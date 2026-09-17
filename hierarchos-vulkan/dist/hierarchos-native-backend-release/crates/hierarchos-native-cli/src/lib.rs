use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env,
    ffi::OsString,
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use hierarchos_inference::{
    initialize_model_package, HierarchosModel, NativeBootstrapConfig, RuntimeState, Sampler,
    SamplingConfig,
};
use hierarchos_vulkan::{
    merge_hierarchos_lora_safetensors, AdamWHyperParams, VulkanDevice, VulkanSeq2SeqTransformer,
    VulkanTransformer, VulkanTransformerArchitecture, VulkanTransformerConfig,
    VulkanTransformerGenerationConfig, VulkanTransformerLoraConfig,
};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Row,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const TOKENIZER_ASSETS: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "generation_config.json",
];

const HF_MODEL_REQUIRED_ASSETS: &[&str] = &[
    "model.safetensors",
    "hierarchos_rust_config.json",
    "hierarchos_config.json",
    "tokenizer.json",
];

const HF_TRANSFORMER_REQUIRED_ASSETS: &[&str] = &["config.json", "tokenizer.json"];
const HF_TRANSFORMER_WEIGHTS: &str = "model.safetensors";
const HF_TRANSFORMER_WEIGHTS_INDEX: &str = "model.safetensors.index.json";

#[derive(Clone, Copy, Debug)]
enum HuggingFaceRepoKind {
    Model,
    Dataset,
}

impl HuggingFaceRepoKind {
    fn url_prefix(self) -> &'static str {
        match self {
            Self::Model => "",
            Self::Dataset => "datasets/",
        }
    }

    fn cache_label(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
        }
    }
}

/// Parameter-efficient native fine-tuning profile for coherent-v9. These are
/// existing learned low-rank/shared-factor tensors in the canonical model, not
/// an injected framework adapter. The resulting package therefore remains a
/// normal model.safetensors artifact with no merge step or runtime dependency.
const NATIVE_FINETUNE_PREFIXES: &[&str] = &[
    "h_rnn.w1",
    "h_rnn.w2",
    "h_rnn.a1",
    "h_rnn.a2",
    "h_rnn.g1",
    "h_rnn.g2",
    "l_rnn.w1",
    "l_rnn.w2",
    "l_rnn.a1",
    "l_rnn.a2",
    "l_rnn.g1",
    "l_rnn.g2",
    "h_deepembed_adapter",
    "l_deepembed_adapter",
    "rosa_adapter",
    "rosa_router",
    "ltm_router",
    "ltm.keys",
    "ltm.vals",
];

pub fn main_entry() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<u8, String> {
    let mut argv: VecDeque<OsString> = env::args_os().skip(1).collect();
    let Some(mode) = argv.pop_front() else {
        print_help();
        return Ok(0);
    };
    let mode = mode.to_string_lossy().into_owned();
    if mode == "-h" || mode == "--help" || mode == "help" {
        print_help();
        return Ok(0);
    }
    match mode.as_str() {
        "train" => run_training(mode.as_str(), argv),
        "finetune" => run_training(mode.as_str(), argv),
        "transformer-train" | "transformer-finetune" => run_transformer_training(mode.as_str(), argv),
        "transformer-generate" | "transformer-infer" | "generate" | "infer" => {
            run_transformer_generate(argv)
        }
        "chat" => run_chat(argv),
        "benchmark" => run_benchmark(argv),
        "devices" => run_devices(argv),
        "doctor" => run_doctor(argv),
        "architectures" | "models" => run_architectures(argv),
        "pull" => run_hf_pull(argv),
        "quantize" => {
            eprintln!(
                "ERROR: Quantized export is intentionally disabled for coherent-v9 because the legacy scalar-RWKV format cannot preserve the current learned function."
            );
            eprintln!(
                "       The scalar-RWKV quantizer cannot preserve DeepEmbed, ROSA, hard ACT, or matrix-state recurrence."
            );
            Ok(2)
        }
        "ckpt-2-inf" => run_ckpt_to_inference(argv),
        "merge-lora" => run_merge_lora(argv),
        other => Err(format!(
            "unknown mode {other:?}; expected train, finetune, transformer-train, transformer-finetune, transformer-generate (aliases: transformer-infer, generate, infer), chat, benchmark, devices, doctor, architectures, pull, quantize, merge-lora, or ckpt-2-inf"
        )),
    }
}

fn print_help() {
    println!(
        "Hierarchos native CLI — Rust inference + all-Vulkan training\n\n\
Usage:\n\
  hierarchos-native-cli train [--model-path MODEL] --train DATA.jsonl --out-dir OUT [options]\n\
  hierarchos-native-cli finetune --model-path MODEL --train DATA.jsonl --out-dir OUT [options]\n\
  hierarchos-native-cli transformer-train --model-path MODEL --train DATA.jsonl --out-dir OUT [options]\n\
  hierarchos-native-cli transformer-finetune --model-path MODEL --train DATA.jsonl --out-dir OUT [--lora-rank N] [options]\n\
  hierarchos-native-cli transformer-generate --model-path MODEL --prompt TEXT [generation options]\n\
  hierarchos-native-cli infer --model-path MODEL --prompt TEXT [generation options]  # shorthand\n\
  hierarchos-native-cli chat --model-path MODEL [--prompt TEXT] [sampling options]\n\
  hierarchos-native-cli benchmark --model-path MODEL [--prompt TEXT] [--benchmark-iterations N]\n\
  hierarchos-native-cli devices\n\
  hierarchos-native-cli doctor [--json]\n\
  hierarchos-native-cli architectures [--json]\n\
  hierarchos-native-cli pull --repo OWNER/REPO --out-dir MODEL [--revision REV]\n\
  hierarchos-native-cli ckpt-2-inf --ckpt-input PACKAGE --inf-output OUT\n\
  hierarchos-native-cli merge-lora --model-path MODEL --lora-adapter-path ADAPTER --out-dir OUT\n\
  hierarchos-native-cli quantize ...\n\n\
Native full-model training accepts the legacy CLI's common aliases (--batch_size, --accumulation-steps,\n\
--starting-lr, --rwkv-weight-decay, --adamw-eps, --training-chunk-size, etc.) and\n\
translates them to hierarchos-vulkan-train. Local raw JSONL is tokenized natively;\n\
schema-v6 token-cache directories and tokenized JSONL pass straight through. Hugging Face\n\
model packages, tokenizer assets, and JSONL dataset files can be fetched directly by this\n\
Rust executable with --hf-model, --hf-tokenizer, and --hf-dataset/--hf-dataset-file.\n\
The Transformer commands use the native architecture registry as their source of truth. Run\n\
`hierarchos-native-cli architectures` for the complete canonical model_type list and wrapper aliases;\n\
the registry currently contains {} canonical architectures and {} package aliases. Unsupported\n\
model types fail closed instead of silently falling back to approximate math.\n\
Transformer generation consumes Hugging Face generation_config.json when present and supports\n\
greedy decoding, sampling/logits warpers, beam search, sequence constraints, and the native\n\
generation KV cache through transformer-generate/transformer-infer.\n\
Raw prompt/completion JSONL uses root-compatible schema discovery, EOS termination,\n\
response-preserving truncation, --min-response-tokens, and the native\n\
--assistant-recovery SFT preset.\n\
Fresh `train` may omit --model-path when --tokenizer-path points to local tokenizer\n\
assets; the CLI then constructs a coherent-v9 SafeTensors package entirely in Rust\n\
before launching the Vulkan trainer. `finetune` still requires an existing model.\n\
`finetune` freezes the full-model optimizer to Hierarchos' built-in low-rank/shared\n\
adapter factors by default; repeat --trainable-prefix to supply an explicit native\n\
selection. It emits a complete canonical model package, so no framework adapter\n\
merge or runtime shim is required. SafeTensors remain canonical FP32-master checkpoints\n\
readable by Rust, Vulkan, and\n\
external CUDA consumers that implement the same tensor contract. This executable\n\
contains no Python/PyTorch compatibility dispatcher. Standard bound Hierarchos PEFT-LoRA\n\
SafeTensors adapters can be merged entirely in Rust; coherent-v9 scalar quantized export\n\
remains explicitly unavailable because it cannot preserve the learned function."
    ,
        VulkanTransformerArchitecture::ALL.len(),
        VulkanTransformerArchitecture::HF_MODEL_TYPE_ALIASES.len()
    );
}

#[derive(Clone, Debug)]
struct TextDatasetOptions {
    text_column: Option<String>,
    prompt_column: Option<String>,
    completion_column: Option<String>,
    alpaca: bool,
    kayla: bool,
    train_prompt_tokens: bool,
    prompt_loss_weight: f32,
    response_loss_weight: f32,
    response_boundary_loss_weight: f32,
    response_boundary_tokens: usize,
    min_response_tokens: usize,
    drop_empty_completions: bool,
    max_length: usize,
}

impl Default for TextDatasetOptions {
    fn default() -> Self {
        Self {
            text_column: None,
            prompt_column: None,
            completion_column: None,
            alpaca: false,
            kayla: false,
            train_prompt_tokens: true,
            prompt_loss_weight: 1.0,
            response_loss_weight: 1.0,
            response_boundary_loss_weight: 1.0,
            response_boundary_tokens: 0,
            min_response_tokens: 1,
            drop_empty_completions: true,
            max_length: 1024,
        }
    }
}

type ArchitectureContractOverrides = BTreeMap<String, Value>;

fn capture_contract_number(
    overrides: &mut ArchitectureContractOverrides,
    field: &str,
    raw: &str,
    option: &str,
) -> Result<(), String> {
    let value = raw
        .parse::<f64>()
        .map_err(|error| format!("invalid {option} value {raw:?}: {error}"))?;
    if !value.is_finite() {
        return Err(format!("{option} must be finite"));
    }
    let number = serde_json::Number::from_f64(value)
        .ok_or_else(|| format!("{option} cannot be represented as a JSON number"))?;
    overrides.insert(field.to_string(), Value::Number(number));
    Ok(())
}

fn capture_contract_integer(
    overrides: &mut ArchitectureContractOverrides,
    field: &str,
    raw: &str,
    option: &str,
) -> Result<(), String> {
    let value = raw
        .parse::<u64>()
        .map_err(|error| format!("invalid {option} value {raw:?}: {error}"))?;
    overrides.insert(field.to_string(), Value::Number(value.into()));
    Ok(())
}

fn architecture_contract_digest(
    contract: &serde_json::Map<String, Value>,
) -> Result<String, String> {
    // serde_json's default Map representation is key-sorted, matching the
    // canonical sort_keys=True/separators=(",", ":") contract used by the
    // CUDA/Python exporter. Contract keys/enum values are ASCII by design.
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| format!("could not serialize architecture contract: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn apply_architecture_contract_overrides_to_config(
    path: &Path,
    overrides: &ArchitectureContractOverrides,
) -> Result<String, String> {
    let mut document: Value = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("could not read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("could not decode {}: {error}", path.display()))?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| format!("{} must contain a top-level JSON object", path.display()))?;
    let digest = {
        let contract = object
            .get_mut("architecture_contract")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                format!(
                    "{} has no canonical architecture_contract object; refusing to create an unverifiable cross-runtime package",
                    path.display()
                )
            })?;
        for (field, value) in overrides {
            contract.insert(field.clone(), value.clone());
        }
        architecture_contract_digest(contract)?
    };
    for (field, value) in overrides {
        object.insert(field.clone(), value.clone());
    }
    object.insert(
        "architecture_contract_sha256".to_string(),
        Value::String(digest.clone()),
    );
    let mut bytes = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("could not encode {}: {error}", path.display()))?;
    bytes.push(b'\n');
    fs::write(path, bytes)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    Ok(digest)
}

fn apply_flat_contract_overrides_to_config(
    path: &Path,
    overrides: &ArchitectureContractOverrides,
    digest: &str,
) -> Result<(), String> {
    let mut document: Value = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("could not read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("could not decode {}: {error}", path.display()))?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| format!("{} must contain a top-level JSON object", path.display()))?;
    for (field, value) in overrides {
        object.insert(field.clone(), value.clone());
    }
    object.insert(
        "architecture_contract_sha256".to_string(),
        Value::String(digest.to_string()),
    );
    let mut bytes = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("could not encode {}: {error}", path.display()))?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| format!("could not write {}: {error}", path.display()))
}

fn config_has_nested_architecture_contract(path: &Path) -> Result<bool, String> {
    let document: Value = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("could not read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("could not decode {}: {error}", path.display()))?;
    Ok(document
        .get("architecture_contract")
        .is_some_and(Value::is_object))
}

fn apply_architecture_contract_overrides_to_package(
    package_dir: &Path,
    overrides: &ArchitectureContractOverrides,
) -> Result<Option<String>, String> {
    if overrides.is_empty() {
        return Ok(None);
    }
    let rust_config = package_dir.join("hierarchos_rust_config.json");
    let digest = apply_architecture_contract_overrides_to_config(&rust_config, overrides)?;

    let compatibility_config = package_dir.join("hierarchos_config.json");
    if config_has_nested_architecture_contract(&compatibility_config)? {
        let compatibility_digest =
            apply_architecture_contract_overrides_to_config(&compatibility_config, overrides)?;
        if compatibility_digest != digest {
            return Err(format!(
                "native config files resolved different architecture contract hashes: {digest} vs {compatibility_digest}"
            ));
        }
    } else {
        apply_flat_contract_overrides_to_config(&compatibility_config, overrides, &digest)?;
    }

    let optional_config = package_dir.join("config.json");
    if optional_config.is_file() {
        if config_has_nested_architecture_contract(&optional_config)? {
            let optional_digest =
                apply_architecture_contract_overrides_to_config(&optional_config, overrides)?;
            if optional_digest != digest {
                return Err(format!(
                    "optional config.json resolved architecture contract hash {optional_digest}, expected {digest}"
                ));
            }
        } else {
            apply_flat_contract_overrides_to_config(&optional_config, overrides, &digest)?;
        }
    }
    hierarchos_inference::ModelConfig::from_model_dir(package_dir)
        .map_err(|error| format!("native contract override is invalid: {error}"))?;
    Ok(Some(digest))
}

fn contract_values_equivalent(saved: &Value, requested: &Value) -> bool {
    if saved == requested {
        return true;
    }
    match (saved.as_f64(), requested.as_f64()) {
        (Some(saved), Some(requested)) => {
            saved.is_finite()
                && requested.is_finite()
                && (saved == requested || (saved as f32).to_bits() == (requested as f32).to_bits())
        }
        _ => false,
    }
}

fn validate_exact_resume_contract_overrides(
    source: &Path,
    overrides: &ArchitectureContractOverrides,
) -> Result<(), String> {
    if overrides.is_empty() {
        return Ok(());
    }
    let package_dir = native_model_package_dir(source)?;
    let config_path = package_dir.join("hierarchos_rust_config.json");
    let document: Value = serde_json::from_slice(
        &fs::read(&config_path)
            .map_err(|error| format!("could not read {}: {error}", config_path.display()))?,
    )
    .map_err(|error| format!("could not decode {}: {error}", config_path.display()))?;
    let contract = document
        .get("architecture_contract")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!(
                "{} has no canonical architecture_contract object",
                config_path.display()
            )
        })?;
    for (field, requested) in overrides {
        let saved = contract.get(field).ok_or_else(|| {
            format!(
                "exact native resume package does not record architecture contract field {field:?}; use --model for a weights-only continuation when changing the training contract"
            )
        })?;
        if !contract_values_equivalent(saved, requested) {
            return Err(format!(
                "exact native resume forbids changing architecture contract field {field:?}: checkpoint={saved}, requested={requested}; use --model for a deliberate weights-only continuation"
            ));
        }
    }
    Ok(())
}

fn link_or_copy_model(source: &Path, destination: &Path) -> Result<(), String> {
    if fs::hard_link(source, destination).is_ok() {
        return Ok(());
    }
    fs::copy(source, destination)
        .map(|_| ())
        .map_err(|error| format!("could not stage {}: {error}", source.display()))
}

fn stage_model_with_contract_overrides(
    source: &Path,
    output_dir: &Path,
    overrides: &ArchitectureContractOverrides,
) -> Result<PathBuf, String> {
    let source = native_model_package_dir(source)?;
    let staging = native_contract_staging_dir(output_dir)?;
    fs::create_dir_all(&staging)
        .map_err(|error| format!("could not create {}: {error}", staging.display()))?;
    let result = (|| {
        copy_native_package_assets(&source, &staging, false)?;
        let model = source.join("model.safetensors");
        if !model.is_file() {
            return Err(format!("source package is missing {}", model.display()));
        }
        link_or_copy_model(&model, &staging.join("model.safetensors"))?;
        let optional_config = staging.join("config.json");
        let source_optional_config = source.join("config.json");
        if source_optional_config.is_file() {
            fs::copy(&source_optional_config, &optional_config).map_err(|error| {
                format!(
                    "could not copy {}: {error}",
                    source_optional_config.display()
                )
            })?;
        }
        apply_architecture_contract_overrides_to_package(&staging, overrides)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    Ok(staging)
}

/// Defaults exposed by the root `hierarchos_cli.py` training surface.
///
/// The low-level Vulkan trainer intentionally has conservative smoke-friendly
/// defaults (batch=1, epoch=1, seed=0). The higher-level native CLI is the
/// compatibility surface, so it supplies the same ordinary training defaults
/// as the root CLI before appending user arguments. The Vulkan parser is
/// last-write-wins, which lets explicit native/legacy flags override these
/// values without a second configuration layer.
fn root_cli_training_defaults() -> Vec<OsString> {
    [
        ("--epochs", "3"),
        ("--batch-size", "64"),
        ("--gradient-accumulation-steps", "1"),
        ("--lr", "1e-4"),
        ("--min-lr", "1e-6"),
        ("--warmup-steps", "0"),
        ("--warmup-ratio", "0"),
        ("--grad-clip", "1.0"),
        ("--eps", "1e-8"),
        ("--weight-decay", "0.1"),
        ("--ponder-loss-weight", "0.01"),
        ("--commitment-loss-weight", "0.5"),
        ("--max-ce-loss-for-backward", "0"),
        ("--max-ponder-cost-for-backward", "0"),
        ("--max-commitment-cost-for-backward", "2"),
        ("--max-skipped-train-batches", "0"),
        ("--save-steps", "0"),
        ("--seed", "1337"),
    ]
    .into_iter()
    .flat_map(|(key, value)| [OsString::from(key), OsString::from(value)])
    .collect()
}

fn run_training(mode: &str, mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let explicit_options = argv
        .iter()
        .filter_map(|arg| {
            let value = arg.to_string_lossy();
            value.starts_with("--").then(|| value.into_owned())
        })
        .collect::<Vec<_>>();
    let mut model_path: Option<PathBuf> = None;
    let mut resume_path: Option<PathBuf> = None;
    let mut dataset_path: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut hf_model: Option<String> = None;
    let mut hf_model_revision = "main".to_string();
    let mut hf_tokenizer: Option<String> = None;
    let mut hf_tokenizer_revision = "main".to_string();
    let mut hf_dataset: Option<String> = None;
    let mut hf_dataset_file: Option<String> = None;
    let mut hf_dataset_config: Option<String> = None;
    let mut hf_dataset_split = "train".to_string();
    let mut hf_dataset_revision = "main".to_string();
    let mut hf_cache_dir: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("./hierarchos_model");
    let mut text_options = TextDatasetOptions::default();
    let mut assistant_recovery = false;
    let mut trainer_args = root_cli_training_defaults();
    let mut contract_overrides = ArchitectureContractOverrides::new();
    let mut unsupported = Vec::new();
    let mut explicit_trainable_prefixes = Vec::<String>::new();
    let mut ignored_peft_geometry = Vec::<String>::new();
    let mut bootstrap_context_dim = 448usize;
    let mut bootstrap_persistent_dim = 128usize;
    let mut bootstrap_ltm_slots = 1024usize;
    let mut bootstrap_ltm_key_dim = 128usize;
    let mut bootstrap_ltm_val_dim = 128usize;
    let mut bootstrap_ltm_topk = 4usize;
    let mut bootstrap_h_hidden = None::<usize>;
    let mut bootstrap_l_hidden = None::<usize>;
    let mut bootstrap_h_stride = 4usize;
    let mut bootstrap_max_h_steps = 5usize;
    let mut bootstrap_max_l_steps = 5usize;
    let mut bootstrap_min_h_steps = 1usize;
    let mut bootstrap_rwkv_head_size = None::<usize>;
    let mut bootstrap_token_adapter_rank = None::<usize>;
    let mut bootstrap_rosa_max_context = 512usize;
    let mut bootstrap_memory_gate_warmup_steps = 2000usize;
    let mut bootstrap_memory_gate_warmup_floor = 0.10f32;
    let mut bootstrap_training_chunk_size = 256usize;
    let mut bootstrap_seed = 1337u64;

    while let Some(arg) = argv.pop_front() {
        let key = arg.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                print_training_help(mode);
                return Ok(0);
            }
            "--model-path" | "--model" => {
                model_path = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--resume-from-ckpt" => {
                resume_path = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--train" => dataset_path = Some(PathBuf::from(required(&mut argv, &key)?)),
            "--out-dir" | "--output" => {
                output_dir = PathBuf::from(required(&mut argv, &key)?);
            }
            "--tokenizer-path" | "--tokenizer_path" => {
                tokenizer_path = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--hf-model" | "--hf_model" => {
                hf_model = Some(required_string(&mut argv, &key)?);
            }
            "--hf-model-revision" | "--hf_model_revision" => {
                hf_model_revision = required_string(&mut argv, &key)?;
            }
            "--hf-tokenizer" | "--hf_tokenizer" => {
                hf_tokenizer = Some(required_string(&mut argv, &key)?);
            }
            "--hf-tokenizer-revision" | "--hf_tokenizer_revision" => {
                hf_tokenizer_revision = required_string(&mut argv, &key)?;
            }
            "--hf-cache-dir" | "--hf_cache_dir" => {
                hf_cache_dir = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--architecture-revision" | "--architecture_revision" => {
                let value = required_string(&mut argv, &key)?;
                if !matches!(
                    value.as_str(),
                    "coherent-v9" | "coherent" | "v9" | "v9-coherent"
                ) {
                    return Err(format!(
                        "native fresh bootstrap supports coherent-v9 only; {key}={value:?} is not a native training contract"
                    ));
                }
            }
            "--context_dim" | "--context-dim" => {
                bootstrap_context_dim = parse_required(&mut argv, &key)?;
            }
            "--persistent_dim" | "--persistent-dim" => {
                bootstrap_persistent_dim = parse_required(&mut argv, &key)?;
            }
            "--ltm_slots" | "--ltm-slots" => {
                bootstrap_ltm_slots = parse_required(&mut argv, &key)?;
            }
            "--ltm_key_dim" | "--ltm-key-dim" => {
                bootstrap_ltm_key_dim = parse_required(&mut argv, &key)?;
            }
            "--ltm_val_dim" | "--ltm-val-dim" => {
                bootstrap_ltm_val_dim = parse_required(&mut argv, &key)?;
            }
            "--ltm_topk" | "--ltm-topk" => {
                bootstrap_ltm_topk = parse_required(&mut argv, &key)?;
            }
            "--h_hidden" | "--h-hidden" => {
                bootstrap_h_hidden = Some(parse_required(&mut argv, &key)?);
            }
            "--l_hidden" | "--l-hidden" => {
                bootstrap_l_hidden = Some(parse_required(&mut argv, &key)?);
            }
            "--h_stride" | "--h-stride" => {
                bootstrap_h_stride = parse_required(&mut argv, &key)?;
            }
            "--max_h_steps" | "--max-h-steps" => {
                bootstrap_max_h_steps = parse_required(&mut argv, &key)?;
            }
            "--max_l_steps" | "--max-l-steps" => {
                bootstrap_max_l_steps = parse_required(&mut argv, &key)?;
            }
            "--min-h-steps" | "--min_h_steps" => {
                bootstrap_min_h_steps = parse_required(&mut argv, &key)?;
            }
            "--rwkv-head-size" | "--rwkv_head_size" => {
                bootstrap_rwkv_head_size = Some(parse_required(&mut argv, &key)?);
            }
            "--token-adapter-rank" | "--token_adapter_rank" => {
                bootstrap_token_adapter_rank = Some(parse_required(&mut argv, &key)?);
            }
            "--rosa-max-context" | "--rosa_max_context" => {
                bootstrap_rosa_max_context = parse_required(&mut argv, &key)?;
            }
            "--memory-gate-warmup-steps" | "--memory_gate_warmup_steps" => {
                let value = required_string(&mut argv, &key)?;
                bootstrap_memory_gate_warmup_steps = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {key} value {value:?}: {error}"))?;
                capture_contract_integer(
                    &mut contract_overrides,
                    "memory_gate_warmup_steps",
                    &value,
                    &key,
                )?;
            }
            "--memory-gate-warmup-floor" | "--memory_gate_warmup_floor" => {
                let value = required_string(&mut argv, &key)?;
                bootstrap_memory_gate_warmup_floor = value
                    .parse::<f32>()
                    .map_err(|error| format!("invalid {key} value {value:?}: {error}"))?;
                capture_contract_number(
                    &mut contract_overrides,
                    "memory_gate_warmup_floor",
                    &value,
                    &key,
                )?;
            }
            "--h_halt_thresh" | "--h-halt-thresh" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(&mut contract_overrides, "h_halt_thresh", &value, &key)?;
            }
            "--l_conv_atol" | "--l-conv-atol" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(&mut contract_overrides, "l_conv_atol", &value, &key)?;
            }
            "--commitment-threshold" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "commitment_threshold",
                    &value,
                    &key,
                )?;
            }
            "--act-depth-temperature" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "act_depth_temperature",
                    &value,
                    &key,
                )?;
            }
            "--halt-logit-clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(&mut contract_overrides, "halt_logit_clamp", &value, &key)?;
            }
            "--recurrent-state-clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "recurrent_state_clamp",
                    &value,
                    &key,
                )?;
            }
            "--context-state-clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "context_state_clamp",
                    &value,
                    &key,
                )?;
            }
            "--drift-state-clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "drift_state_clamp",
                    &value,
                    &key,
                )?;
            }
            "--drift-norm-clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(&mut contract_overrides, "drift_norm_clamp", &value, &key)?;
            }
            "--drift-delta-scale" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "drift_delta_scale",
                    &value,
                    &key,
                )?;
            }
            "--activation-clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(&mut contract_overrides, "activation_clamp", &value, &key)?;
            }
            "--rwkv-channel-mix-key-clamp" | "--rwkv_channel_mix_key_clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "rwkv_channel_mix_key_clamp",
                    &value,
                    &key,
                )?;
            }
            "--rwkv-channel-mix-deepembed-clamp" | "--rwkv_channel_mix_deepembed_clamp" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "rwkv_channel_mix_deepembed_clamp",
                    &value,
                    &key,
                )?;
            }
            "--ltm-score-grad-scale" | "--ltm_score_grad_scale" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "ltm_score_grad_scale",
                    &value,
                    &key,
                )?;
            }
            "--ltm-value-alignment-weight" | "--ltm_value_alignment_weight" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "ltm_value_alignment_weight",
                    &value,
                    &key,
                )?;
            }
            "--ltm-value-alignment-stride" | "--ltm_value_alignment_stride" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_integer(
                    &mut contract_overrides,
                    "ltm_value_alignment_stride",
                    &value,
                    &key,
                )?;
            }
            "--ltm-value-alignment-min-updates" | "--ltm_value_alignment_min_updates" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_integer(
                    &mut contract_overrides,
                    "ltm_value_alignment_min_updates",
                    &value,
                    &key,
                )?;
            }
            "--ltm-value-alignment-ready-threshold" | "--ltm_value_alignment_ready_threshold" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "ltm_value_alignment_ready_threshold",
                    &value,
                    &key,
                )?;
            }
            "--ltm-value-alignment-ema-decay" | "--ltm_value_alignment_ema_decay" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "ltm_value_alignment_ema_decay",
                    &value,
                    &key,
                )?;
            }
            "--ltm-value-writer-max-norm" | "--ltm_value_writer_max_norm" => {
                let value = required_string(&mut argv, &key)?;
                capture_contract_number(
                    &mut contract_overrides,
                    "ltm_value_writer_max_norm",
                    &value,
                    &key,
                )?;
            }
            "--use-deepembed"
            | "--use-rosa"
            | "--enforce-rosa-max-context"
            | "--rosa-zero-no-prediction"
            | "--memory-token-routers" => {}
            "--no-deepembed"
            | "--no-rosa"
            | "--no-enforce-rosa-max-context"
            | "--no-rosa-zero-no-prediction"
            | "--no-memory-token-routers" => {
                return Err(format!(
                    "{key} is an ablation outside the supported coherent-v9 native runtime contract"
                ));
            }
            "--deepembed-mode" => {
                let value = required_string(&mut argv, &key)?;
                if value != "shared-factorized" {
                    return Err(format!(
                        "native coherent-v9 requires {key}=shared-factorized"
                    ));
                }
            }
            "--rosa-embedding-mode" => {
                let value = required_string(&mut argv, &key)?;
                if value != "shared-factorized" {
                    return Err(format!(
                        "native coherent-v9 requires {key}=shared-factorized"
                    ));
                }
            }
            "--core-recurrence-version" => {
                let value: usize = parse_required(&mut argv, &key)?;
                if value != 2 {
                    return Err(
                        "native coherent-v9 requires --core-recurrence-version 2".to_string()
                    );
                }
            }
            "--manager-compute-mode" => {
                let value = required_string(&mut argv, &key)?;
                if value != "hard-masked" {
                    return Err(
                        "native coherent-v9 requires --manager-compute-mode hard-masked"
                            .to_string(),
                    );
                }
            }
            "--hf_dataset" | "--hf-dataset" => {
                hf_dataset = Some(required_string(&mut argv, &key)?);
            }
            "--hf-dataset-file" | "--hf_dataset_file" => {
                hf_dataset_file = Some(required_string(&mut argv, &key)?);
            }
            "--hf-dataset-revision" | "--hf_dataset_revision" => {
                hf_dataset_revision = required_string(&mut argv, &key)?;
            }
            "--hf_dataset_config" | "--hf-dataset-config" => {
                hf_dataset_config = Some(required_string(&mut argv, &key)?);
            }
            "--hf_dataset_split" | "--hf-dataset-split" => {
                hf_dataset_split = required_string(&mut argv, &key)?;
            }
            "--text_column" | "--text-column" => {
                text_options.text_column = Some(required_string(&mut argv, &key)?);
            }
            "--prompt_column" | "--prompt-column" => {
                text_options.prompt_column = Some(required_string(&mut argv, &key)?);
            }
            "--completion_column" | "--completion-column" => {
                text_options.completion_column = Some(required_string(&mut argv, &key)?);
            }
            "--alpaca" => text_options.alpaca = true,
            "--kayla" => text_options.kayla = true,
            "--train-prompt-tokens" => text_options.train_prompt_tokens = true,
            "--mask-prompt-tokens" => text_options.train_prompt_tokens = false,
            "--prompt-loss-weight" => {
                text_options.prompt_loss_weight = parse_required(&mut argv, &key)?;
            }
            "--response-loss-weight" => {
                text_options.response_loss_weight = parse_required(&mut argv, &key)?;
            }
            "--response-boundary-loss-weight" => {
                text_options.response_boundary_loss_weight = parse_required(&mut argv, &key)?;
            }
            "--response-boundary-tokens" => {
                text_options.response_boundary_tokens = parse_required(&mut argv, &key)?;
            }
            "--min-response-tokens" | "--min_response_tokens" => {
                text_options.min_response_tokens = parse_required(&mut argv, &key)?;
            }
            "--allow-empty-completions" => text_options.drop_empty_completions = false,
            "--assistant-recovery" | "--assistant_recovery" => assistant_recovery = true,
            "--max_length" | "--max-length" => {
                text_options.max_length = parse_required(&mut argv, &key)?;
            }
            "--batch_size" | "--batch-size" => {
                forward_value(&mut trainer_args, &mut argv, "--batch-size", &key)?
            }
            "--accumulation-steps" | "--accumulation_steps" | "--gradient-accumulation-steps" => {
                forward_value(
                    &mut trainer_args,
                    &mut argv,
                    "--gradient-accumulation-steps",
                    &key,
                )?
            }
            "--starting-lr" | "--lr" => forward_value(&mut trainer_args, &mut argv, "--lr", &key)?,
            "--min-lr" => forward_value(&mut trainer_args, &mut argv, "--min-lr", &key)?,
            "--warmup-steps" | "--warmup_steps" => {
                forward_value(&mut trainer_args, &mut argv, "--warmup-steps", &key)?
            }
            "--warmup-ratio" | "--warmup_ratio" => {
                forward_value(&mut trainer_args, &mut argv, "--warmup-ratio", &key)?
            }
            "--adamw-eps" | "--adamw_eps" | "--eps" => {
                forward_value(&mut trainer_args, &mut argv, "--eps", &key)?
            }
            "--rwkv-weight-decay" | "--rwkv_weight_decay" | "--weight-decay" => {
                forward_value(&mut trainer_args, &mut argv, "--weight-decay", &key)?
            }
            "--training-chunk-size" | "--training_chunk_size" | "--tbptt-chunk-size" => {
                let value = required_string(&mut argv, &key)?;
                bootstrap_training_chunk_size = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid value for {key}: {error}"))?;
                capture_contract_integer(
                    &mut contract_overrides,
                    "training_chunk_size",
                    &value,
                    &key,
                )?;
                capture_contract_integer(
                    &mut contract_overrides,
                    "reference_chunk_len",
                    &value,
                    &key,
                )?;
                trainer_args.push(OsString::from("--tbptt-chunk-size"));
                trainer_args.push(OsString::from(value));
            }
            "--trainable-prefix" | "--finetune-target-prefix" => {
                explicit_trainable_prefixes.push(required_string(&mut argv, &key)?);
            }
            "--z-loss-weight"
            | "--ponder-loss-weight"
            | "--commitment-loss-weight"
            | "--max-ponder-cost-for-backward"
            | "--max-commitment-cost-for-backward" => {
                let value = required_string(&mut argv, &key)?;
                let field = key.trim_start_matches("--").replace('-', "_");
                capture_contract_number(&mut contract_overrides, &field, &value, &key)?;
                trainer_args.push(OsString::from(&key));
                trainer_args.push(OsString::from(value));
            }
            "--epochs"
            | "--grad-clip"
            | "--initial-loss-scale"
            | "--precision"
            | "--beta1"
            | "--beta2"
            | "--max-ce-loss-for-backward"
            | "--max-skipped-train-batches"
            | "--save-steps"
            | "--device-index"
            | "--device-indices"
            | "--gradient-stream-chunk-values"
            | "--joint-runtime-profile" => forward_value(&mut trainer_args, &mut argv, &key, &key)?,
            "--seed" => {
                let value = required_string(&mut argv, &key)?;
                bootstrap_seed = value
                    .parse::<u64>()
                    .map_err(|error| format!("invalid value for {key}: {error}"))?;
                trainer_args.push(OsString::from("--seed"));
                trainer_args.push(OsString::from(value));
            }
            "--amp" => {
                trainer_args.push(OsString::from("--precision"));
                trainer_args.push(OsString::from("fp16-storage-parity"));
            }
            "--no-amp" | "--no_amp" => {
                trainer_args.push(OsString::from("--precision"));
                trainer_args.push(OsString::from("fp32"));
            }
            "--accumulation-normalization" => {
                let value = required_string(&mut argv, &key)?;
                if value != "weighted-token" {
                    return Err(format!(
                        "native Vulkan training implements the canonical weighted-token accumulation objective; {key}={value:?} is not parity-safe"
                    ));
                }
            }
            "--ltm-training-mode" | "--ltm_training_mode" => {
                let value = required_string(&mut argv, &key)?;
                if value != "read-only" {
                    return Err(format!(
                        "native coherent-v9 training currently implements inference-like/read-only LTM fast-memory semantics; {key}={value:?} is not supported"
                    ));
                }
            }
            "--inference-like-ltm-training" => {}
            "--disable-lr-schedule"
            | "--persist-state"
            | "--no-shuffle"
            | "--json-events"
            | "--lock-joint-runtime-profile" => trainer_args.push(arg),
            "--no-persist-state" => {}
            // Accepted compatibility no-ops: these affect the legacy DataLoader/autograd path rather than
            // the native training graph, whose equivalent policy is intrinsic or autotuned.
            "--full-sample-bptt"
            | "--full_sample_bptt"
            | "--no-full-sample-bptt"
            | "--no_full_sample_bptt"
            | "--full-sample-activation-checkpointing"
            | "--full_sample_activation_checkpointing"
            | "--no-full-sample-activation-checkpointing"
            | "--no_full_sample_activation_checkpointing"
            | "--gradient-checkpointing"
            | "--gradient_checkpointing"
            | "--compile"
            | "--force-compile"
            | "--no-compile"
            | "--no_compile"
            | "--hf-token-cache"
            | "--local-token-cache"
            | "--length-bucketing"
            | "--no-length-bucketing"
            | "--auto-length-bucket-size"
            | "--no-auto-length-bucket-size" => {}
            "--lora_r" | "--lora_alpha" | "--lora_dropout" | "--finetune-unlock-percent" => {
                let value = required_string(&mut argv, &key)?;
                if mode != "finetune" {
                    return Err(format!("{key} is only valid with native finetune"));
                }
                ignored_peft_geometry.push(format!("{key}={value}"));
            }
            // Values that are meaningful only to the legacy loader/evaluator are consumed so users
            // can reuse launch scripts without positional drift, then reported once.
            "--num-workers"
            | "--num_workers"
            | "--prefetch-factor"
            | "--prefetch_factor"
            | "--dataset-size"
            | "--progress-log-steps"
            | "--progress_log_steps"
            | "--padding-metric-steps"
            | "--padding_metric_steps"
            | "--max-sanitized-gradient-values" => {
                let value = required(&mut argv, &key)?;
                unsupported.push(format!("{key}={}", value.to_string_lossy()));
            }
            "--no-padding-metrics" => {}
            "--pre_chunked_dataset" | "--pre-chunked-dataset" => {}
            "--pre_pt_dataset" | "--pre-pt-dataset" => {
                return Err(
                    "native training does not load framework-specific .pt dataset objects; use tokenized JSONL or a schema-v6 token cache"
                        .to_string(),
                );
            }
            "--eval-batch-size"
            | "--eval-limit"
            | "--eval-every-epoch"
            | "--best-checkpoint-metric" => {
                return Err(format!(
                    "{key} is not implemented by the pure-native training frontend"
                ));
            }
            other => {
                return Err(format!(
                    "native Vulkan train does not implement {other:?}; refusing to fall back to a non-native runtime"
                ));
            }
        }
    }

    let explicit = |aliases: &[&str]| {
        aliases
            .iter()
            .any(|alias| explicit_options.iter().any(|option| option == alias))
    };
    if assistant_recovery {
        if mode != "train" {
            eprintln!(
                "warning: --assistant-recovery only affects native train mode; ignoring preset"
            );
        } else {
            if text_options.kayla {
                return Err(
                    "--assistant-recovery targets Alpaca instruction/input/output data and cannot be combined with --kayla"
                        .to_string(),
                );
            }
            if !explicit(&["--alpaca"]) {
                text_options.alpaca = true;
            }
            if !explicit(&["--epochs"]) {
                trainer_args.push(OsString::from("--epochs"));
                trainer_args.push(OsString::from("4"));
            }
            if !explicit(&["--starting-lr", "--lr"]) {
                trainer_args.push(OsString::from("--lr"));
                trainer_args.push(OsString::from("6e-5"));
            }
            if !explicit(&["--min-lr"]) {
                trainer_args.push(OsString::from("--min-lr"));
                trainer_args.push(OsString::from("1e-6"));
            }
            if !explicit(&["--warmup-ratio", "--warmup_ratio"]) {
                trainer_args.push(OsString::from("--warmup-ratio"));
                trainer_args.push(OsString::from("0.03"));
            }
            if !explicit(&["--prompt-loss-weight"]) {
                text_options.prompt_loss_weight = 0.10;
            }
            if !explicit(&["--response-loss-weight"]) {
                text_options.response_loss_weight = 1.0;
            }
            if !explicit(&["--response-boundary-loss-weight"]) {
                text_options.response_boundary_loss_weight = 2.0;
            }
            if !explicit(&["--response-boundary-tokens"]) {
                text_options.response_boundary_tokens = 32;
            }
            if !explicit(&["--min-response-tokens", "--min_response_tokens"]) {
                text_options.min_response_tokens = 16;
            }
            if !explicit(&["--ponder-loss-weight"]) {
                trainer_args.push(OsString::from("--ponder-loss-weight"));
                trainer_args.push(OsString::from("0.003"));
                capture_contract_number(
                    &mut contract_overrides,
                    "ponder_loss_weight",
                    "0.003",
                    "--assistant-recovery",
                )?;
            }
            if !explicit(&["--memory-gate-warmup-steps", "--memory_gate_warmup_steps"]) {
                bootstrap_memory_gate_warmup_steps = 5000;
                capture_contract_integer(
                    &mut contract_overrides,
                    "memory_gate_warmup_steps",
                    "5000",
                    "--assistant-recovery",
                )?;
            }
            eprintln!(
                "native assistant-recovery preset: alpaca={} epochs=4 lr=6e-5 warmup_ratio=0.03 prompt_weight={} response_boundary={}x{} min_response_tokens={} (explicit CLI values override)",
                text_options.alpaca,
                text_options.prompt_loss_weight,
                text_options.response_boundary_tokens,
                text_options.response_boundary_loss_weight,
                text_options.min_response_tokens,
            );
        }
    }

    if !unsupported.is_empty() {
        eprintln!(
            "warning: ignored legacy loader-only options: {}",
            unsupported.join(", ")
        );
    }
    if !ignored_peft_geometry.is_empty() {
        eprintln!(
            "warning: native finetune uses model-defined coherent-v9 low-rank/shared-factor geometry; legacy PEFT geometry options are not architecture mutations and were ignored: {}",
            ignored_peft_geometry.join(", ")
        );
    }

    // Match the root CLI contract where --tokenizer-path may be either a local
    // path or a Hugging Face repo id. Existing local paths always win; a
    // missing OWNER/REPO-shaped value is resolved through the native Hub path.
    if hf_tokenizer.is_none() {
        let tokenizer_repo = tokenizer_path.as_ref().and_then(|path| {
            if path.exists() {
                return None;
            }
            let candidate = path.to_string_lossy().into_owned();
            validate_hf_repo_id(&candidate).ok().map(|()| candidate)
        });
        if let Some(repo) = tokenizer_repo {
            hf_tokenizer = Some(repo);
            tokenizer_path = None;
        }
    }

    if hf_model.is_some() && (model_path.is_some() || resume_path.is_some()) {
        return Err(
            "--hf-model is mutually exclusive with --model-path/--model and --resume-from-ckpt"
                .to_string(),
        );
    }
    if model_path.is_some() && resume_path.is_some() {
        return Err(
            "--model-path/--model and --resume-from-ckpt are mutually exclusive: use --model for a weights-only new run or --resume-from-ckpt for exact native continuation"
                .to_string(),
        );
    }
    if hf_tokenizer.is_some() && tokenizer_path.is_some() {
        return Err("--hf-tokenizer is mutually exclusive with --tokenizer-path".to_string());
    }
    if dataset_path.is_some() && hf_dataset.is_some() {
        return Err("--train and --hf-dataset are mutually exclusive".to_string());
    }

    let uses_hf = hf_model.is_some() || hf_tokenizer.is_some() || hf_dataset.is_some();
    let hf_cache = if uses_hf {
        Some(resolve_hf_cache_root(hf_cache_dir.as_deref())?)
    } else {
        None
    };
    if let Some(repo) = hf_model.as_deref() {
        let cache = hf_cache
            .as_deref()
            .expect("Hugging Face cache must exist when --hf-model is set");
        let package = fetch_hf_model_package(repo, &hf_model_revision, cache)?;
        eprintln!(
            "native Hugging Face model: {repo}@{hf_model_revision} -> {}",
            package.display()
        );
        if tokenizer_path.is_none() {
            tokenizer_path = Some(package.clone());
        }
        model_path = Some(package);
    }
    if let Some(repo) = hf_tokenizer.as_deref() {
        let cache = hf_cache
            .as_deref()
            .expect("Hugging Face cache must exist when --hf-tokenizer is set");
        let tokenizer_dir = fetch_hf_tokenizer_assets(repo, &hf_tokenizer_revision, cache)?;
        eprintln!(
            "native Hugging Face tokenizer: {repo}@{hf_tokenizer_revision} -> {}",
            tokenizer_dir.display()
        );
        tokenizer_path = Some(tokenizer_dir);
    }

    let dataset = match (dataset_path, hf_dataset.as_deref()) {
        (Some(path), None) => path,
        (None, Some(repo)) => {
            let cache = hf_cache
                .as_deref()
                .expect("Hugging Face cache must exist when --hf-dataset is set");
            let remote_files = if let Some(remote_file) = hf_dataset_file.as_deref() {
                vec![remote_file.to_string()]
            } else {
                let discovered = discover_hf_dataset_files(
                    repo,
                    &hf_dataset_revision,
                    hf_dataset_config.as_deref(),
                    &hf_dataset_split,
                )?;
                eprintln!(
                    "native Hugging Face dataset discovery: {repo}@{hf_dataset_revision} config={:?} split={} -> {} file(s)",
                    hf_dataset_config,
                    hf_dataset_split,
                    discovered.len()
                );
                discovered
            };
            let path = fetch_hf_dataset_files(
                repo,
                &hf_dataset_revision,
                &remote_files,
                cache,
                hf_dataset_config.as_deref(),
                &hf_dataset_split,
            )?;
            eprintln!(
                "native Hugging Face dataset: {repo}@{hf_dataset_revision} -> {}",
                path.display()
            );
            path
        }
        (None, None) => {
            return Err(format!(
                "native {mode} requires --train DATA or --hf-dataset OWNER/REPO"
            ))
        }
        (Some(_), Some(_)) => unreachable!("mutual exclusion validated above"),
    };
    let trainable_prefixes = if mode == "finetune" && explicit_trainable_prefixes.is_empty() {
        NATIVE_FINETUNE_PREFIXES
            .iter()
            .map(|prefix| (*prefix).to_string())
            .collect::<Vec<_>>()
    } else {
        explicit_trainable_prefixes
    };
    for prefix in trainable_prefixes {
        trainer_args.push(OsString::from("--trainable-prefix"));
        trainer_args.push(OsString::from(prefix));
    }
    let mut bootstrap_dir = None::<PathBuf>;
    let mut contract_dir = None::<PathBuf>;
    let source_model = if let Some(source) = resume_path.as_ref() {
        // Exact resume is content/identity bound. Never rewrite its package in
        // place or construct a partial staged copy that omits optimizer/replay
        // state. Repeated root-CLI contract flags are accepted only when they
        // exactly describe the saved package.
        validate_exact_resume_contract_overrides(source, &contract_overrides)?;
        source.clone()
    } else if let Some(source) = model_path.as_ref() {
        if contract_overrides.is_empty() {
            source.clone()
        } else {
            let staging =
                stage_model_with_contract_overrides(source, &output_dir, &contract_overrides)?;
            eprintln!(
                "native contract staging: bound {} root-CLI training/architecture override(s) into {}",
                contract_overrides.len(),
                staging.display()
            );
            contract_dir = Some(staging.clone());
            staging
        }
    } else {
        if mode != "train" {
            return Err(
                "native finetune requires --model-path MODEL or --resume-from-ckpt PACKAGE"
                    .to_string(),
            );
        }
        let tokenizer_source = tokenizer_path.as_ref().ok_or_else(|| {
            "fresh native train requires --tokenizer-path TOKENIZER so vocabulary geometry can be constructed without Python"
                .to_string()
        })?;
        let tokenizer_file = resolve_local_tokenizer_path(tokenizer_source)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|error| format!("could not load {}: {error}", tokenizer_file.display()))?;
        let vocab_size = tokenizer.get_vocab_size(true);
        if vocab_size == 0 {
            return Err("fresh native train tokenizer has an empty vocabulary".to_string());
        }
        let mut bootstrap = NativeBootstrapConfig::for_vocab(vocab_size);
        bootstrap.context_dim = bootstrap_context_dim;
        bootstrap.persistent_dim = bootstrap_persistent_dim;
        bootstrap.ltm_slots = bootstrap_ltm_slots;
        bootstrap.ltm_key_dim = bootstrap_ltm_key_dim;
        bootstrap.ltm_val_dim = bootstrap_ltm_val_dim;
        bootstrap.ltm_topk = bootstrap_ltm_topk;
        bootstrap.h_hidden = bootstrap_h_hidden.unwrap_or(bootstrap_context_dim);
        bootstrap.l_hidden = bootstrap_l_hidden.unwrap_or(bootstrap_context_dim);
        bootstrap.h_stride = bootstrap_h_stride;
        bootstrap.max_h_steps = bootstrap_max_h_steps;
        bootstrap.max_l_steps = bootstrap_max_l_steps;
        bootstrap.min_h_steps = bootstrap_min_h_steps;
        bootstrap.token_adapter_rank =
            bootstrap_token_adapter_rank.unwrap_or_else(|| bootstrap_context_dim.min(64));
        bootstrap.rosa_max_context = bootstrap_rosa_max_context;
        bootstrap.memory_gate_warmup_steps = bootstrap_memory_gate_warmup_steps;
        bootstrap.memory_gate_warmup_floor = bootstrap_memory_gate_warmup_floor;
        bootstrap.training_chunk_size = bootstrap_training_chunk_size;
        bootstrap.seed = bootstrap_seed;
        bootstrap.resolve_auto_geometry(bootstrap_rwkv_head_size);
        let staging = native_bootstrap_staging_dir(&output_dir)?;
        initialize_model_package(&staging, &bootstrap)
            .map_err(|error| format!("native model bootstrap failed: {error}"))?;
        if let Some(digest) =
            apply_architecture_contract_overrides_to_package(&staging, &contract_overrides)?
        {
            eprintln!(
                "native bootstrap contract: applied {} explicit/preset override(s), architecture_contract_sha256={digest}",
                contract_overrides.len()
            );
        }
        install_local_tokenizer_assets(tokenizer_source, &staging)?;
        eprintln!(
            "native bootstrap: initialized coherent-v9 model from scratch at {} (vocab={}, context={}, h={}, l={}, head={})",
            staging.display(),
            vocab_size,
            bootstrap.context_dim,
            bootstrap.h_hidden,
            bootstrap.l_hidden,
            bootstrap.rwkv_head_size
        );
        bootstrap_dir = Some(staging.clone());
        staging
    };
    let prepared_dataset = prepare_training_dataset(
        &dataset,
        &source_model,
        tokenizer_path.as_deref(),
        &output_dir,
        &text_options,
    )?;

    let trainer = find_vulkan_executable("hierarchos-vulkan-train").ok_or_else(|| {
        "could not locate hierarchos-vulkan-train; build hierarchos-vulkan first".to_string()
    })?;
    let mut command = Command::new(&trainer);
    if resume_path.is_some() {
        command.arg("--resume-from-ckpt").arg(&source_model);
    } else {
        command.arg("--model").arg(&source_model);
    }
    command.arg("--dataset").arg(prepared_dataset);
    command.arg("--output").arg(&output_dir);
    command.args(trainer_args);
    command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let status = command
        .status()
        .map_err(|error| format!("failed to launch {}: {error}", trainer.display()))?;
    let code = status.code().unwrap_or(1).clamp(0, 255) as u8;
    if code == 0 {
        if let Some(tokenizer_path) = tokenizer_path.as_deref() {
            install_local_tokenizer_assets(tokenizer_path, &output_dir)?;
            install_local_tokenizer_assets_into_checkpoints(tokenizer_path, &output_dir)?;
        }
        if let Some(staging) = bootstrap_dir.as_ref() {
            fs::remove_dir_all(staging).map_err(|error| {
                format!(
                    "could not remove native bootstrap staging {}: {error}",
                    staging.display()
                )
            })?;
        }
        if let Some(staging) = contract_dir.as_ref() {
            fs::remove_dir_all(staging).map_err(|error| {
                format!(
                    "could not remove native contract staging {}: {error}",
                    staging.display()
                )
            })?;
        }
    } else {
        if let Some(staging) = bootstrap_dir.as_ref() {
            eprintln!(
                "native bootstrap staging retained after trainer failure for inspection: {}",
                staging.display()
            );
        }
        if let Some(staging) = contract_dir.as_ref() {
            eprintln!(
                "native contract staging retained after trainer failure for inspection: {}",
                staging.display()
            );
        }
    }
    Ok(code)
}

fn native_bootstrap_staging_dir(output_dir: &Path) -> Result<PathBuf, String> {
    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let output_name = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hierarchos-model");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before UNIX_EPOCH: {error}"))?
        .as_nanos();
    let path = parent.join(format!(
        ".{output_name}.native-bootstrap-{}-{nonce}",
        std::process::id()
    ));
    if path.exists() {
        return Err(format!(
            "native bootstrap staging path already exists: {}",
            path.display()
        ));
    }
    Ok(path)
}

fn native_contract_staging_dir(output_dir: &Path) -> Result<PathBuf, String> {
    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let output_name = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hierarchos-model");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before UNIX_EPOCH: {error}"))?
        .as_nanos();
    let path = parent.join(format!(
        ".{output_name}.native-contract-{}-{nonce}",
        std::process::id()
    ));
    if path.exists() {
        return Err(format!(
            "native contract staging path already exists: {}",
            path.display()
        ));
    }
    Ok(path)
}

fn print_training_help(mode: &str) {
    println!(
        "hierarchos-native-cli {mode} [--model-path MODEL] --train DATA --out-dir OUT [options]\n\n\
Fresh train may omit --model-path when --tokenizer-path is supplied. In that case the CLI builds\n\
a coherent-v9 model package entirely in Rust, then trains it through the Vulkan backend. finetune\n\
still requires a base model or exact native resume package.\n\n\
Hugging Face is native too: --hf-model OWNER/REPO downloads a canonical model package,\n\
--hf-tokenizer OWNER/REPO supplies tokenizer assets for fresh initialization, and\n\
--tokenizer-path OWNER/REPO is accepted as the root-CLI-compatible tokenizer spelling.\n\
--hf-dataset OWNER/REPO discovers JSONL/NDJSON files for --hf_dataset_split (default train);\n\
--hf-dataset-file PATH.jsonl may be used to select an exact file instead. Downloads are\n\
stored in the local native cache. Use --hf-*-revision to pin revisions and --hf-cache-dir DIR\n\
to relocate the cache. HF_TOKEN or HUGGING_FACE_HUB_TOKEN enables private/gated repos.\n\n\
DATA may be a schema-v6 token-cache directory, tokenized JSONL with input_ids, or raw\n\
local JSONL containing text or prompt/completion fields. Raw JSONL is tokenized with\n\
MODEL/tokenizer.json by default. --tokenizer-path may instead point to a local tokenizer.json\n\
or tokenizer directory; successful native outputs and periodic checkpoints receive those assets.\n\
Raw rows recognize text/content, instruction/output, prompt/completion, and question/answer;\n\
prompt/completion rows append EOS, preserve the prompt suffix plus answer prefix when truncated,\n\
drop blank completions by default, and enforce --min-response-tokens (default 1). Use\n\
--allow-empty-completions only for intentional EOS-only answers.\n\n\
Common legacy aliases: --batch_size, --accumulation-steps, --starting-lr, --min-lr,\n\
--warmup-steps, --warmup-ratio, --rwkv-weight-decay, --adamw-eps, --grad-clip,\n\
--training-chunk-size, --persist-state, --seed, --alpaca, --mask-prompt-tokens,\n\
--prompt-loss-weight, --response-loss-weight, --response-boundary-loss-weight,\n\
--response-boundary-tokens, --min-response-tokens, and --assistant-recovery.\n\
The native assistant-recovery preset enables Alpaca formatting and supported root defaults\n\
(4 epochs, 6e-5 LR, 0.03 warmup ratio, prompt/response weighting, 2x first 32 response\n\
tokens, 16 reserved answer tokens, 0.003 ponder weight, and 5000-step fresh-model memory-gate\n\
warmup); explicit CLI values still win. Framework-only LTM optimizer/objective controls are\n\
not synthesized by this frontend and continue to fail closed rather than invoking Python.\n\n\
Fresh-model geometry aliases include --context_dim, --persistent_dim, --ltm_slots,\n\
--ltm_key_dim, --ltm_val_dim, --ltm_topk, --h_hidden, --l_hidden, --h_stride,\n\
--max_h_steps, --max_l_steps, --min-h-steps, --rwkv-head-size, --token-adapter-rank,\n\
--rosa-max-context, --memory-gate-warmup-steps, and --memory-gate-warmup-floor.\n\n\
Root-CLI defaults are preserved here (epochs=3, batch_size=64, seed=1337, min_lr=1e-6,\n\
ponder_loss_weight=0.01). --amp maps to the qualified Vulkan fp16-storage-parity\n\
policy and --no-amp maps to fp32.\n\n\
Native extensions: --precision POLICY, --device-index N, --device-indices N,N,...,\n\
--gradient-stream-chunk-values N, --joint-runtime-profile PATH, --lock-joint-runtime-profile,\n\
--trainable-prefix PREFIX (repeatable). Native finetune defaults to coherent-v9's built-in\n\
low-rank recurrent factors, DeepEmbed/ROSA adapters, routers, and slow LTM tensors and emits\n\
a complete SafeTensors model package directly."
    );
}

fn run_hf_pull(mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let mut repo = None::<String>;
    let mut revision = "main".to_string();
    let mut output = None::<PathBuf>;
    let mut cache_dir = None::<PathBuf>;
    let mut overwrite = false;
    while let Some(arg) = argv.pop_front() {
        let key = arg.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                println!(
                    "hierarchos-native-cli pull --repo OWNER/REPO --out-dir MODEL [--revision REV] [--hf-cache-dir DIR] [--overwrite]\n\nDownloads a canonical Hierarchos SafeTensors/config/tokenizer package directly from the Hugging Face Hub using Rust HTTPS. HF_TOKEN or HUGGING_FACE_HUB_TOKEN is used when present. No Python, PyTorch, git-lfs, or huggingface_hub Python package is launched."
                );
                return Ok(0);
            }
            "--repo" | "--hf-model" | "--hf_model" => {
                repo = Some(required_string(&mut argv, &key)?);
            }
            "--revision" | "--hf-model-revision" | "--hf_model_revision" => {
                revision = required_string(&mut argv, &key)?;
            }
            "--out-dir" | "--output" => {
                output = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--hf-cache-dir" | "--hf_cache_dir" => {
                cache_dir = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--overwrite" => overwrite = true,
            other => return Err(format!("unsupported pull argument {other:?}")),
        }
    }
    let repo = repo.ok_or("pull requires --repo OWNER/REPO")?;
    let output = output.ok_or("pull requires --out-dir MODEL")?;
    let cache = resolve_hf_cache_root(cache_dir.as_deref())?;
    let source = fetch_hf_model_package(&repo, &revision, &cache)?;
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock cannot create pull staging name: {error}"))?
        .as_nanos();
    let output_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hierarchos-model");
    let staging = parent.join(format!(
        ".{output_name}.native-hf-pull-{}-{nonce}",
        std::process::id()
    ));
    let result = (|| -> Result<(), String> {
        fs::create_dir(&staging)
            .map_err(|error| format!("could not create {}: {error}", staging.display()))?;
        copy_native_package_assets(&source, &staging, true)?;
        let tokenizer_path = staging.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
        validate_tokenizer_vocab_for_package(&tokenizer, &staging, &tokenizer_path)?;
        // Do not publish a Hub pull until the full canonical tensor/config
        // contract is consumable by the framework-free inference runtime.
        let _ = NativeSession::load(&staging)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    publish_staged_directory(&staging, &output, overwrite)?;
    println!(
        "native Hugging Face pull complete: {repo}@{revision} -> {}",
        output.display()
    );
    Ok(0)
}

fn resolve_hf_cache_root(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let path = explicit
        .map(Path::to_path_buf)
        .or_else(|| env::var_os("HIERARCHOS_HF_CACHE").map(PathBuf::from))
        .unwrap_or_else(|| repo_root().join(".hierarchos-hf-cache"));
    fs::create_dir_all(&path).map_err(|error| {
        format!(
            "could not create Hugging Face cache {}: {error}",
            path.display()
        )
    })?;
    Ok(path)
}

#[derive(Debug)]
struct TransformerChunk {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
    encoder_context: Option<Arc<TransformerEncoderContext>>,
}

#[derive(Debug)]
struct TransformerEncoderContext {
    hidden_states: Vec<f32>,
    attention_mask: Vec<f32>,
    seq_len: usize,
}

fn disable_transformer_dropout_fields(config: &mut Value) -> usize {
    const ROOT_DROPOUT_FIELDS: &[&str] = &[
        "resid_pdrop",
        "embd_pdrop",
        "attn_pdrop",
        "hidden_dropout",
        "attention_dropout",
        "resid_dropout",
        "embed_dropout",
        "residual_dropout",
        "embedding_dropout",
        "emb_pdrop",
        "dropout",
        "layerdrop",
    ];

    let Some(root) = config.as_object_mut() else {
        return 0;
    };
    let mut changed = 0usize;
    for field in ROOT_DROPOUT_FIELDS {
        let Some(value) = root.get_mut(*field) else {
            continue;
        };
        if value.as_f64().is_some_and(|probability| probability != 0.0) {
            *value = json!(0.0);
            changed += 1;
        }
    }
    if let Some(attn_config) = root.get_mut("attn_config").and_then(Value::as_object_mut) {
        if let Some(value) = attn_config.get_mut("attn_pdrop") {
            if value.as_f64().is_some_and(|probability| probability != 0.0) {
                *value = json!(0.0);
                changed += 1;
            }
        }
    }
    changed
}

fn run_transformer_training(mode: &str, mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let mut model_path: Option<PathBuf> = None;
    let mut hf_model: Option<String> = None;
    let mut hf_model_revision = "main".to_owned();
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut hf_tokenizer: Option<String> = None;
    let mut hf_tokenizer_revision = "main".to_owned();
    let mut dataset_path: Option<PathBuf> = None;
    let mut hf_dataset: Option<String> = None;
    let mut hf_dataset_file: Option<String> = None;
    let mut hf_dataset_config: Option<String> = None;
    let mut hf_dataset_split = "train".to_owned();
    let mut hf_dataset_revision = "main".to_owned();
    let mut hf_cache_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut batch_size = 1usize;
    let mut seq_len = 128usize;
    let mut epochs = 1usize;
    let mut device_index: Option<usize> = None;
    let mut lr = 5.0e-5f32;
    let mut beta1 = 0.9f32;
    let mut beta2 = 0.999f32;
    let mut eps = 1.0e-8f32;
    let mut weight_decay = 0.01f32;
    let mut lora_rank: Option<usize> = None;
    let mut lora_alpha: Option<usize> = None;
    let mut lora_dropout = 0.0f64;
    let mut lora_bias = "none".to_owned();
    let mut lora_targets = Vec::<String>::new();
    let mut lora_modules_to_save = Vec::<String>::new();
    let mut lora_use_dora = false;
    let mut lora_use_rslora = false;
    let mut lora_adapter_path: Option<PathBuf> = None;
    let mut seed = 42u64;
    let mut disable_transformer_dropout = false;
    let mut text_options = TextDatasetOptions::default();

    while let Some(raw) = argv.pop_front() {
        let key = raw.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                println!(
                    "hierarchos-native-cli {mode} (--model-path DIR | --hf-model OWNER/REPO) (--train DATA.jsonl | --hf-dataset OWNER/REPO) --out-dir OUT [--batch-size N] [--seq-len N] [--epochs N] [--lr F] [--device-index N] [--seed N] [--lora-rank N] [--lora-alpha N] [--lora-dropout F] [--lora-bias none|lora_only|all] [--lora-target-module NAME ...] [--lora-modules-to-save NAME ...] [--lora-use-dora] [--lora-use-rslora] [--lora-adapter-path DIR] [--disable-transformer-dropout]\n\nThe model math, stochastic dropout, loss, backpropagation and AdamW step execute in Vulkan. Hugging Face model/dataset/tokenizer I/O, Parquet-to-JSONL conversion, and tokenization execute in Rust; Python and PyTorch are not launched. Supported checkpoint dropout probabilities are honored natively using reproducible counter-based RNG derived from --seed. --disable-transformer-dropout remains an explicit ablation/debug override that zeroes recognized dropout fields for this run and records those zero values on full-model export. Pretokenized cross-attention decoder rows may provide encoder_hidden_states (flat or nested [encoder_sequence, hidden]) and optional encoder_attention_mask; encoder geometry must remain fixed across the dataset."
                );
                return Ok(0);
            }
            "--model-path" | "--model_path" | "--model" => {
                model_path = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--hf-model" | "--hf_model" => hf_model = Some(required_string(&mut argv, &key)?),
            "--hf-model-revision" | "--hf_model_revision" => {
                hf_model_revision = required_string(&mut argv, &key)?
            }
            "--tokenizer-path" | "--tokenizer_path" => {
                tokenizer_path = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--hf-tokenizer" | "--hf_tokenizer" => {
                hf_tokenizer = Some(required_string(&mut argv, &key)?)
            }
            "--hf-tokenizer-revision" | "--hf_tokenizer_revision" => {
                hf_tokenizer_revision = required_string(&mut argv, &key)?
            }
            "--train" | "--dataset" | "--data" => {
                dataset_path = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--hf-dataset" | "--hf_dataset" => hf_dataset = Some(required_string(&mut argv, &key)?),
            "--hf-dataset-file" | "--hf_dataset_file" => {
                hf_dataset_file = Some(required_string(&mut argv, &key)?)
            }
            "--hf-dataset-config" | "--hf_dataset_config" => {
                hf_dataset_config = Some(required_string(&mut argv, &key)?)
            }
            "--hf-dataset-split" | "--hf_dataset_split" => {
                hf_dataset_split = required_string(&mut argv, &key)?
            }
            "--hf-dataset-revision" | "--hf_dataset_revision" => {
                hf_dataset_revision = required_string(&mut argv, &key)?
            }
            "--hf-cache-dir" | "--hf_cache_dir" => {
                hf_cache_dir = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--out-dir" | "--out_dir" | "--output" => {
                output_dir = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--batch-size" | "--batch_size" => batch_size = parse_required(&mut argv, &key)?,
            "--seq-len" | "--seq_len" | "--max-length" | "--max_length" => {
                seq_len = parse_required(&mut argv, &key)?
            }
            "--epochs" => epochs = parse_required(&mut argv, &key)?,
            "--device-index" | "--device_index" => {
                device_index = Some(parse_required(&mut argv, &key)?)
            }
            "--lr" | "--learning-rate" | "--learning_rate" | "--starting-lr" => {
                lr = parse_required(&mut argv, &key)?
            }
            "--beta1" | "--adam-beta1" => beta1 = parse_required(&mut argv, &key)?,
            "--beta2" | "--adam-beta2" => beta2 = parse_required(&mut argv, &key)?,
            "--eps" | "--adamw-eps" => eps = parse_required(&mut argv, &key)?,
            "--weight-decay" | "--weight_decay" => weight_decay = parse_required(&mut argv, &key)?,
            "--lora-rank" | "--lora_rank" => lora_rank = Some(parse_required(&mut argv, &key)?),
            "--lora-alpha" | "--lora_alpha" => lora_alpha = Some(parse_required(&mut argv, &key)?),
            "--lora-dropout" | "--lora_dropout" => lora_dropout = parse_required(&mut argv, &key)?,
            "--lora-bias" | "--lora_bias" => lora_bias = required_string(&mut argv, &key)?,
            "--lora-target-module" | "--lora_target_module" => {
                lora_targets.push(required_string(&mut argv, &key)?)
            }
            "--lora-modules-to-save" | "--lora_modules_to_save" => {
                lora_modules_to_save.push(required_string(&mut argv, &key)?)
            }
            "--lora-use-dora" | "--lora_use_dora" | "--use-dora" | "--use_dora" => {
                lora_use_dora = true
            }
            "--lora-use-rslora" | "--lora_use_rslora" | "--use-rslora" | "--use_rslora" => {
                lora_use_rslora = true
            }
            "--lora-adapter-path" | "--lora_adapter_path" | "--peft-adapter" => {
                lora_adapter_path = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--seed" => seed = parse_required(&mut argv, &key)?,
            "--disable-transformer-dropout" | "--disable_transformer_dropout" => {
                disable_transformer_dropout = true
            }
            "--text-column" | "--text_column" => {
                text_options.text_column = Some(required_string(&mut argv, &key)?)
            }
            "--prompt-column" | "--prompt_column" => {
                text_options.prompt_column = Some(required_string(&mut argv, &key)?)
            }
            "--completion-column" | "--completion_column" => {
                text_options.completion_column = Some(required_string(&mut argv, &key)?)
            }
            "--alpaca" => text_options.alpaca = true,
            "--train-prompt-tokens" | "--train_prompt_tokens" => {
                text_options.train_prompt_tokens = true
            }
            "--no-train-prompt-tokens" | "--no_train_prompt_tokens" => {
                text_options.train_prompt_tokens = false
            }
            "--prompt-loss-weight" | "--prompt_loss_weight" => {
                text_options.prompt_loss_weight = parse_required(&mut argv, &key)?
            }
            "--response-loss-weight" | "--response_loss_weight" => {
                text_options.response_loss_weight = parse_required(&mut argv, &key)?
            }
            other => return Err(format!("unknown {mode} option {other:?}")),
        }
    }

    if batch_size == 0 || seq_len == 0 || epochs == 0 {
        return Err(
            "Transformer batch size, sequence length, and epochs must be positive".to_owned(),
        );
    }
    if model_path.is_some() == hf_model.is_some() {
        return Err(format!(
            "{mode} requires exactly one of --model-path or --hf-model"
        ));
    }
    if dataset_path.is_some() == hf_dataset.is_some() {
        return Err(format!(
            "{mode} requires exactly one of --train or --hf-dataset"
        ));
    }
    if tokenizer_path.is_some() && hf_tokenizer.is_some() {
        return Err("--tokenizer-path and --hf-tokenizer are mutually exclusive".to_owned());
    }
    if lora_adapter_path.is_some() && lora_rank.is_some() {
        return Err("--lora-adapter-path and --lora-rank are mutually exclusive".to_owned());
    }
    if lora_adapter_path.is_some()
        && (lora_dropout != 0.0
            || lora_bias != "none"
            || !lora_targets.is_empty()
            || !lora_modules_to_save.is_empty()
            || lora_use_dora
            || lora_use_rslora)
    {
        return Err(
            "PEFT configuration flags cannot be combined with --lora-adapter-path; resumed adapters use adapter_config.json"
                .to_owned(),
        );
    }
    let output_dir = output_dir.ok_or_else(|| format!("{mode} requires --out-dir"))?;
    let uses_hf = hf_model.is_some() || hf_tokenizer.is_some() || hf_dataset.is_some();
    let hf_cache = if uses_hf {
        Some(resolve_hf_cache_root(hf_cache_dir.as_deref())?)
    } else {
        None
    };

    let base_model_label = hf_model
        .clone()
        .or_else(|| model_path.as_ref().map(|path| path.display().to_string()));
    let model_dir = if let Some(repo) = hf_model.as_deref() {
        fetch_hf_transformer_package(
            repo,
            &hf_model_revision,
            hf_cache.as_deref().expect("HF cache"),
            disable_transformer_dropout,
        )?
    } else {
        model_path.expect("validated model path")
    };
    let tokenizer_override = if let Some(repo) = hf_tokenizer.as_deref() {
        Some(fetch_hf_tokenizer_assets(
            repo,
            &hf_tokenizer_revision,
            hf_cache.as_deref().expect("HF cache"),
        )?)
    } else {
        tokenizer_path
    };
    let dataset = if let Some(repo) = hf_dataset.as_deref() {
        let cache = hf_cache.as_deref().expect("HF cache");
        let remote_files = if let Some(file) = hf_dataset_file.as_deref() {
            vec![file.to_owned()]
        } else {
            discover_hf_dataset_files(
                repo,
                &hf_dataset_revision,
                hf_dataset_config.as_deref(),
                &hf_dataset_split,
            )?
        };
        fetch_hf_dataset_files(
            repo,
            &hf_dataset_revision,
            &remote_files,
            cache,
            hf_dataset_config.as_deref(),
            &hf_dataset_split,
        )?
    } else {
        dataset_path.expect("validated dataset path")
    };

    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("could not create {}: {error}", output_dir.display()))?;
    text_options.max_length = seq_len.saturating_add(1);
    let prepared = prepare_training_dataset(
        &dataset,
        &model_dir,
        tokenizer_override.as_deref(),
        &output_dir,
        &text_options,
    )?;
    if prepared.is_dir() {
        return Err("Transformer training currently accepts tokenized/raw JSONL, not Hierarchos schema-v6 cache directories".to_owned());
    }

    let source_config_path = model_dir.join("config.json");
    let source_config =
        VulkanTransformerConfig::from_hf_config(&source_config_path).map_err(|error| {
            format!(
                "invalid Hugging Face Transformer package {}: {error:#}",
                source_config_path.display()
            )
        })?;
    let (runtime_config, dropout_config_override) = if disable_transformer_dropout {
        let config_path = model_dir.join("config.json");
        let mut value: Value = serde_json::from_slice(
            &fs::read(&config_path)
                .map_err(|error| format!("could not read {}: {error}", config_path.display()))?,
        )
        .map_err(|error| format!("could not decode {}: {error}", config_path.display()))?;
        let changed = disable_transformer_dropout_fields(&mut value);
        let config = VulkanTransformerConfig::from_hf_value(&value)
            .map_err(|error| format!("invalid Hugging Face Transformer package: {error:#}"))?;
        if changed > 0 {
            eprintln!(
                "native Transformer dropout override: explicitly disabled {changed} configured checkpoint dropout field(s) for this run"
            );
            (config, Some(value))
        } else {
            (config, None)
        }
    } else {
        (source_config, None)
    };

    let encoder_seq_len =
        transformer_encoder_seq_len_from_jsonl(&prepared, runtime_config.hidden_size)?;

    let device = match device_index {
        Some(index) => VulkanDevice::new_with_index(index),
        None => VulkanDevice::new(),
    }
    .map_err(|error| format!("Vulkan device initialization failed: {error:#}"))?;
    let mut graph = match encoder_seq_len {
        Some(encoder_seq_len) => {
            VulkanTransformer::from_hf_package_with_config_and_encoder_seq_len(
                device,
                &model_dir,
                runtime_config,
                batch_size,
                seq_len,
                encoder_seq_len,
            )
        }
        None => VulkanTransformer::from_hf_package_with_config(
            device,
            &model_dir,
            runtime_config,
            batch_size,
            seq_len,
        ),
    }
    .map_err(|error| format!("could not build Vulkan Transformer graph: {error:#}"))?;
    match (graph.encoder_seq_len(), encoder_seq_len) {
        (Some(_), None) => {
            return Err(format!(
                "{} checkpoint contains cross-attention blocks; tokenized training rows must provide encoder_hidden_states (and optionally encoder_attention_mask) so cross-attention cannot be silently skipped",
                graph.config().architecture.model_type()
            ));
        }
        (None, Some(_)) => {
            return Err(format!(
                "dataset provides encoder_hidden_states, but {} has no cross-attention encoder input",
                graph.config().architecture.model_type()
            ));
        }
        _ => {}
    }
    graph.set_dropout_seed((seed as u32) ^ ((seed >> 32) as u32));

    if let Some(adapter_dir) = lora_adapter_path.as_deref() {
        let count = graph.load_lora_adapter(adapter_dir).map_err(|error| {
            format!(
                "could not load PEFT adapter {}: {error:#}",
                adapter_dir.display()
            )
        })?;
        eprintln!(
            "Vulkan PEFT: resumed {count} LoRA module(s) from {}",
            adapter_dir.display()
        );
    } else {
        if mode == "transformer-finetune" && lora_rank.is_none() {
            lora_rank = Some(8);
        }
        if let Some(rank) = lora_rank {
            let targets = if lora_targets.is_empty() {
                vec![graph
                    .config()
                    .architecture
                    .default_lora_target_module()
                    .to_owned()]
            } else {
                lora_targets
            };
            let alpha = lora_alpha.unwrap_or(rank * 2);
            let fan_in_fan_out = matches!(
                graph.config().architecture,
                VulkanTransformerArchitecture::Gpt2 | VulkanTransformerArchitecture::GptSw3
            );
            let mut config =
                VulkanTransformerLoraConfig::causal_lm(rank, alpha, targets, fan_in_fan_out)
                    .map_err(|error| format!("invalid LoRA configuration: {error:#}"))?;
            config.base_model_name_or_path = base_model_label;
            config.lora_dropout = lora_dropout;
            config.bias = lora_bias;
            config.modules_to_save =
                (!lora_modules_to_save.is_empty()).then_some(lora_modules_to_save);
            config.use_dora = lora_use_dora;
            config.use_rslora = lora_use_rslora;
            let count = graph
                .enable_lora(config, seed)
                .map_err(|error| format!("could not enable Vulkan PEFT: {error:#}"))?;
            eprintln!(
                "Vulkan PEFT: attached {count} LoRA module(s); base {:?} parameters are frozen",
                graph.config().architecture
            );
        }
    }

    let hyper = AdamWHyperParams {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    };
    let mut total_steps = 0usize;
    for epoch in 0..epochs {
        let (steps, mean_loss) =
            train_transformer_jsonl_epoch(&mut graph, &prepared, batch_size, seq_len, hyper)?;
        if steps == 0 {
            return Err("Transformer dataset produced no supervised training batches".to_owned());
        }
        total_steps += steps;
        eprintln!(
            "Transformer epoch {}/{}: steps={} mean_loss={:.6}",
            epoch + 1,
            epochs,
            steps,
            mean_loss
        );
    }

    if graph.lora_config().is_some() {
        graph
            .export_lora_adapter(&output_dir)
            .map_err(|error| format!("could not export PEFT adapter: {error:#}"))?;
        if let Some(source) = tokenizer_override.as_deref() {
            install_local_tokenizer_assets(source, &output_dir)?;
        } else if model_dir.join("tokenizer.json").is_file() {
            install_local_tokenizer_assets(&model_dir, &output_dir)?;
        }
    } else {
        graph
            .export_hf_package(&model_dir, &output_dir)
            .map_err(|error| format!("could not export trained Transformer package: {error:#}"))?;
        if let Some(config) = dropout_config_override.as_ref() {
            let config_path = output_dir.join("config.json");
            let encoded = serde_json::to_vec_pretty(config).map_err(|error| {
                format!("could not encode Transformer config with dropout disabled: {error}")
            })?;
            fs::write(&config_path, encoded)
                .map_err(|error| format!("could not write {}: {error}", config_path.display()))?;
        }
    }
    eprintln!(
        "native Transformer training complete: backend=Vulkan device={} steps={} output={}",
        graph.device_name(),
        total_steps,
        output_dir.display()
    );
    Ok(0)
}

fn train_transformer_jsonl_epoch(
    graph: &mut VulkanTransformer,
    path: &Path,
    batch_size: usize,
    seq_len: usize,
    hyper: AdamWHyperParams,
) -> Result<(usize, f64), String> {
    let file =
        File::open(path).map_err(|error| format!("could not open {}: {error}", path.display()))?;
    let mut pending = Vec::<TransformerChunk>::with_capacity(batch_size);
    let mut steps = 0usize;
    let mut loss_sum = 0.0f64;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("could not read {}: {error}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|error| {
            format!("invalid tokenized JSON on line {}: {error}", line_index + 1)
        })?;
        for chunk in transformer_chunks_from_value(&value, seq_len, graph.config().hidden_size)
            .map_err(|error| format!("line {}: {error}", line_index + 1))?
        {
            pending.push(chunk);
            if pending.len() == batch_size {
                let loss = submit_transformer_chunks(graph, &pending, batch_size, seq_len, hyper)?;
                if let Some(loss) = loss {
                    loss_sum += f64::from(loss);
                    steps += 1;
                }
                pending.clear();
            }
        }
    }
    if !pending.is_empty() {
        if let Some(loss) = submit_transformer_chunks(graph, &pending, batch_size, seq_len, hyper)?
        {
            loss_sum += f64::from(loss);
            steps += 1;
        }
    }
    Ok((
        steps,
        if steps == 0 {
            0.0
        } else {
            loss_sum / steps as f64
        },
    ))
}

fn flatten_transformer_f32_tensor(
    value: &Value,
    field: &str,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if let Some(values) = value.as_array() {
        for value in values {
            flatten_transformer_f32_tensor(value, field, output)?;
        }
        return Ok(());
    }
    let raw = value
        .as_f64()
        .ok_or_else(|| format!("{field} must contain only numeric values"))? as f32;
    if !raw.is_finite() {
        return Err(format!("{field} values must be finite"));
    }
    output.push(raw);
    Ok(())
}

fn transformer_encoder_context_from_value(
    value: &Value,
    hidden_size: usize,
) -> Result<Option<Arc<TransformerEncoderContext>>, String> {
    let Some(hidden_value) = value.get("encoder_hidden_states") else {
        if value.get("encoder_attention_mask").is_some() {
            return Err(
                "encoder_attention_mask was supplied without encoder_hidden_states".to_owned(),
            );
        }
        return Ok(None);
    };
    if hidden_size == 0 {
        return Err(
            "Transformer hidden size must be positive for encoder hidden states".to_owned(),
        );
    }
    let mut hidden_states = Vec::new();
    flatten_transformer_f32_tensor(hidden_value, "encoder_hidden_states", &mut hidden_states)?;
    if hidden_states.is_empty() || hidden_states.len() % hidden_size != 0 {
        return Err(format!(
            "encoder_hidden_states must contain a positive multiple of model hidden size {hidden_size}, got {} values",
            hidden_states.len()
        ));
    }
    let seq_len = hidden_states.len() / hidden_size;
    let attention_mask = match value.get("encoder_attention_mask") {
        Some(mask) => mask
            .as_array()
            .ok_or_else(|| "encoder_attention_mask must be a numeric array".to_owned())?
            .iter()
            .map(|value| {
                let mask = value
                    .as_f64()
                    .ok_or_else(|| "encoder_attention_mask must be numeric".to_owned())?
                    as f32;
                if !mask.is_finite() {
                    return Err("encoder_attention_mask values must be finite".to_owned());
                }
                Ok(mask)
            })
            .collect::<Result<Vec<_>, String>>()?,
        None => vec![1.0; seq_len],
    };
    if attention_mask.len() != seq_len {
        return Err(format!(
            "encoder_attention_mask must contain exactly {seq_len} values, got {}",
            attention_mask.len()
        ));
    }
    Ok(Some(Arc::new(TransformerEncoderContext {
        hidden_states,
        attention_mask,
        seq_len,
    })))
}

fn transformer_encoder_seq_len_from_jsonl(
    path: &Path,
    hidden_size: usize,
) -> Result<Option<usize>, String> {
    let file =
        File::open(path).map_err(|error| format!("could not open {}: {error}", path.display()))?;
    let mut expected = None;
    let mut saw_context = false;
    let mut saw_without_context = false;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("could not read {}: {error}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|error| {
            format!("invalid tokenized JSON on line {}: {error}", line_index + 1)
        })?;
        match transformer_encoder_context_from_value(&value, hidden_size)
            .map_err(|error| format!("line {}: {error}", line_index + 1))?
        {
            Some(context) => {
                saw_context = true;
                if let Some(expected) = expected {
                    if context.seq_len != expected {
                        return Err(format!(
                            "line {}: encoder sequence length must stay fixed at {expected}, got {}",
                            line_index + 1,
                            context.seq_len
                        ));
                    }
                } else {
                    expected = Some(context.seq_len);
                }
            }
            None => saw_without_context = true,
        }
    }
    if saw_context && saw_without_context {
        return Err(
            "Transformer dataset mixes rows with encoder_hidden_states and rows without encoder context"
                .to_owned(),
        );
    }
    Ok(expected)
}

fn transformer_chunks_from_value(
    value: &Value,
    seq_len: usize,
    hidden_size: usize,
) -> Result<Vec<TransformerChunk>, String> {
    let encoder_context = transformer_encoder_context_from_value(value, hidden_size)?;
    let ids = value
        .get("input_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "tokenized Transformer row is missing input_ids".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| "input_ids must be u32 integers".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let labels = match value.get("labels").and_then(Value::as_array) {
        Some(values) => values
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| "labels must be integers".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => ids.iter().map(|&id| i64::from(id)).collect(),
    };
    let attention = match value.get("attention_mask").and_then(Value::as_array) {
        Some(values) => values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| "attention_mask must be numeric".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![1.0; ids.len()],
    };
    let weights = match value.get("loss_weights").and_then(Value::as_array) {
        Some(values) => values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| "loss_weights must be numeric".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![1.0; ids.len()],
    };
    let n = ids
        .len()
        .min(labels.len())
        .min(attention.len())
        .min(weights.len());
    if n < 2 {
        return Ok(Vec::new());
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start + 1 < n {
        let count = seq_len.min(n - 1 - start);
        let mut input_ids = vec![0u32; seq_len];
        let mut targets = vec![u32::MAX; seq_len];
        let mut attention_mask = vec![0.0f32; seq_len];
        let mut loss_weights = vec![0.0f32; seq_len];
        for local in 0..count {
            let source = start + local;
            input_ids[local] = ids[source];
            attention_mask[local] = attention[source];
            let next_label = labels[source + 1];
            if attention[source] > 0.0 && attention[source + 1] > 0.0 && next_label >= 0 {
                targets[local] = u32::try_from(next_label)
                    .map_err(|_| "label is outside u32 range".to_owned())?;
                loss_weights[local] = weights[source + 1].max(0.0);
            }
        }
        if loss_weights.iter().any(|&weight| weight > 0.0) {
            chunks.push(TransformerChunk {
                input_ids,
                targets,
                attention_mask,
                loss_weights,
                encoder_context: encoder_context.clone(),
            });
        }
        start += count;
    }
    Ok(chunks)
}

fn submit_transformer_chunks(
    graph: &mut VulkanTransformer,
    chunks: &[TransformerChunk],
    batch_size: usize,
    seq_len: usize,
    hyper: AdamWHyperParams,
) -> Result<Option<f32>, String> {
    let rows = batch_size
        .checked_mul(seq_len)
        .ok_or_else(|| "Transformer batch geometry overflow".to_owned())?;
    let mut input_ids = vec![0u32; rows];
    let mut targets = vec![u32::MAX; rows];
    let mut attention_mask = vec![0.0f32; rows];
    let mut loss_weights = vec![0.0f32; rows];
    let encoder_seq_len = graph.encoder_seq_len();
    let encoder_values_per_sequence = encoder_seq_len
        .map(|encoder_seq_len| {
            encoder_seq_len
                .checked_mul(graph.config().hidden_size)
                .ok_or_else(|| "Transformer encoder hidden geometry overflow".to_owned())
        })
        .transpose()?;
    let mut encoder_hidden_states =
        encoder_values_per_sequence.map(|values| vec![0.0f32; values.saturating_mul(batch_size)]);
    let mut encoder_attention_mask = encoder_seq_len
        .map(|encoder_seq_len| vec![0.0f32; encoder_seq_len.saturating_mul(batch_size)]);
    for (batch, chunk) in chunks.iter().take(batch_size).enumerate() {
        let base = batch * seq_len;
        input_ids[base..base + seq_len].copy_from_slice(&chunk.input_ids);
        targets[base..base + seq_len].copy_from_slice(&chunk.targets);
        attention_mask[base..base + seq_len].copy_from_slice(&chunk.attention_mask);
        loss_weights[base..base + seq_len].copy_from_slice(&chunk.loss_weights);
        match (
            chunk.encoder_context.as_ref(),
            encoder_seq_len,
            encoder_values_per_sequence,
            encoder_hidden_states.as_mut(),
            encoder_attention_mask.as_mut(),
        ) {
            (Some(context), Some(expected_seq_len), Some(values), Some(hidden), Some(mask)) => {
                if context.seq_len != expected_seq_len {
                    return Err(format!(
                        "Transformer batch encoder sequence length changed from {expected_seq_len} to {}",
                        context.seq_len
                    ));
                }
                let hidden_base = batch * values;
                hidden[hidden_base..hidden_base + values].copy_from_slice(&context.hidden_states);
                let mask_base = batch * expected_seq_len;
                mask[mask_base..mask_base + expected_seq_len]
                    .copy_from_slice(&context.attention_mask);
            }
            (None, None, None, None, None) => {}
            (None, Some(_), _, _, _) => {
                return Err(
                    "cross-attention Transformer batch is missing encoder context".to_owned(),
                );
            }
            (Some(_), None, _, _, _) => {
                return Err(
                    "Transformer batch supplies encoder context to a graph without cross-attention"
                        .to_owned(),
                );
            }
            _ => return Err("inconsistent Transformer encoder batch buffers".to_owned()),
        }
    }
    if !loss_weights.iter().any(|&weight| weight > 0.0) {
        return Ok(None);
    }
    let result = match (
        encoder_hidden_states.as_deref(),
        encoder_attention_mask.as_deref(),
    ) {
        (Some(encoder_hidden_states), Some(encoder_attention_mask)) => graph
            .train_step_with_encoder_hidden_states(
                &input_ids,
                &targets,
                &attention_mask,
                &loss_weights,
                encoder_hidden_states,
                encoder_attention_mask,
                hyper,
            ),
        (None, None) => {
            graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)
        }
        _ => unreachable!("encoder hidden states and mask are allocated together"),
    }
    .map_err(|error| format!("Vulkan Transformer step failed: {error:#}"))?;
    Ok(Some(result.loss))
}

fn fetch_hf_model_package(
    repo: &str,
    revision: &str,
    cache_root: &Path,
) -> Result<PathBuf, String> {
    validate_hf_repo_id(repo)?;
    let destination = hf_repo_cache_dir(cache_root, HuggingFaceRepoKind::Model, repo, revision)?;
    fs::create_dir_all(&destination)
        .map_err(|error| format!("could not create {}: {error}", destination.display()))?;
    for name in HF_MODEL_REQUIRED_ASSETS {
        download_hf_file(
            HuggingFaceRepoKind::Model,
            repo,
            revision,
            name,
            &destination.join(name),
            true,
        )?;
    }
    for name in TOKENIZER_ASSETS
        .iter()
        .copied()
        .filter(|name| *name != "tokenizer.json")
    {
        download_hf_file(
            HuggingFaceRepoKind::Model,
            repo,
            revision,
            name,
            &destination.join(name),
            false,
        )?;
    }
    let tokenizer_path = destination.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
    validate_tokenizer_vocab_for_package(&tokenizer, &destination, &tokenizer_path)?;
    Ok(destination)
}

fn fetch_hf_transformer_package(
    repo: &str,
    revision: &str,
    cache_root: &Path,
    disable_transformer_dropout: bool,
) -> Result<PathBuf, String> {
    validate_hf_repo_id(repo)?;
    let destination = hf_repo_cache_dir(cache_root, HuggingFaceRepoKind::Model, repo, revision)?
        .join("transformer-hf");
    fs::create_dir_all(&destination)
        .map_err(|error| format!("could not create {}: {error}", destination.display()))?;
    for name in HF_TRANSFORMER_REQUIRED_ASSETS {
        download_hf_file(
            HuggingFaceRepoKind::Model,
            repo,
            revision,
            name,
            &destination.join(name),
            true,
        )?;
    }
    fetch_hf_transformer_weights(repo, revision, &destination)?;
    for name in TOKENIZER_ASSETS
        .iter()
        .copied()
        .filter(|name| *name != "tokenizer.json")
    {
        download_hf_file(
            HuggingFaceRepoKind::Model,
            repo,
            revision,
            name,
            &destination.join(name),
            false,
        )?;
    }
    let config_path = destination.join("config.json");
    let config = if disable_transformer_dropout {
        let mut value: Value = serde_json::from_slice(
            &fs::read(&config_path)
                .map_err(|error| format!("could not read {}: {error}", config_path.display()))?,
        )
        .map_err(|error| format!("could not decode {}: {error}", config_path.display()))?;
        disable_transformer_dropout_fields(&mut value);
        VulkanTransformerConfig::from_hf_value(&value)
    } else {
        VulkanTransformerConfig::from_hf_config(&config_path)
    }
    .map_err(|error| format!("invalid Hugging Face Transformer package: {error:#}"))?;
    let tokenizer_path = destination.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
    if tokenizer.get_vocab_size(true) > config.vocab_size {
        return Err(format!(
            "Transformer tokenizer vocabulary exceeds model embedding rows for {}: tokenizer={} model={}",
            tokenizer_path.display(),
            tokenizer.get_vocab_size(true),
            config.vocab_size
        ));
    }
    Ok(destination)
}

fn fetch_hf_transformer_weights(
    repo: &str,
    revision: &str,
    destination: &Path,
) -> Result<(), String> {
    if download_hf_file(
        HuggingFaceRepoKind::Model,
        repo,
        revision,
        HF_TRANSFORMER_WEIGHTS,
        &destination.join(HF_TRANSFORMER_WEIGHTS),
        false,
    )? {
        return Ok(());
    }

    let index_path = destination.join(HF_TRANSFORMER_WEIGHTS_INDEX);
    download_hf_file(
        HuggingFaceRepoKind::Model,
        repo,
        revision,
        HF_TRANSFORMER_WEIGHTS_INDEX,
        &index_path,
        true,
    )?;
    let index: Value = serde_json::from_slice(
        &fs::read(&index_path)
            .map_err(|error| format!("could not read {}: {error}", index_path.display()))?,
    )
    .map_err(|error| format!("could not parse {}: {error}", index_path.display()))?;
    for shard in transformer_safetensors_shards_from_index_value(&index)? {
        let local_path = shard
            .split('/')
            .fold(destination.to_path_buf(), |path, segment| {
                path.join(segment)
            });
        download_hf_file(
            HuggingFaceRepoKind::Model,
            repo,
            revision,
            &shard,
            &local_path,
            true,
        )?;
    }
    Ok(())
}

fn transformer_safetensors_shards_from_index_value(value: &Value) -> Result<Vec<String>, String> {
    let weight_map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| "Transformer SafeTensors index is missing object weight_map".to_owned())?;
    if weight_map.is_empty() {
        return Err("Transformer SafeTensors index weight_map must not be empty".to_owned());
    }
    let mut shards = BTreeSet::new();
    for (tensor_name, shard_value) in weight_map {
        let shard = shard_value.as_str().ok_or_else(|| {
            format!("Transformer SafeTensors index entry {tensor_name:?} is not a shard filename")
        })?;
        validate_hf_relative_path(shard)?;
        if shard.contains('\\') || !shard.to_ascii_lowercase().ends_with(".safetensors") {
            return Err(format!(
                "Transformer SafeTensors index entry {tensor_name:?} references non-SafeTensors file {shard:?}"
            ));
        }
        shards.insert(shard.to_owned());
    }
    Ok(shards.into_iter().collect())
}

fn fetch_hf_tokenizer_assets(
    repo: &str,
    revision: &str,
    cache_root: &Path,
) -> Result<PathBuf, String> {
    validate_hf_repo_id(repo)?;
    let destination = hf_repo_cache_dir(cache_root, HuggingFaceRepoKind::Model, repo, revision)?
        .join("tokenizer-only");
    fs::create_dir_all(&destination)
        .map_err(|error| format!("could not create {}: {error}", destination.display()))?;
    for name in TOKENIZER_ASSETS {
        let required = *name == "tokenizer.json";
        download_hf_file(
            HuggingFaceRepoKind::Model,
            repo,
            revision,
            name,
            &destination.join(name),
            required,
        )?;
    }
    let tokenizer_path = destination.join("tokenizer.json");
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
    Ok(destination)
}

fn fetch_hf_dataset_file(
    repo: &str,
    revision: &str,
    remote_file: &str,
    cache_root: &Path,
) -> Result<PathBuf, String> {
    validate_hf_repo_id(repo)?;
    validate_hf_relative_path(remote_file)?;
    if !is_hf_native_dataset_path(remote_file) {
        return Err(format!(
            "native Hugging Face training currently accepts explicit JSONL/NDJSON/Parquet files; {remote_file:?} is not a supported native dataset filename"
        ));
    }
    let destination_dir =
        hf_repo_cache_dir(cache_root, HuggingFaceRepoKind::Dataset, repo, revision)?;
    fs::create_dir_all(&destination_dir)
        .map_err(|error| format!("could not create {}: {error}", destination_dir.display()))?;
    let destination = remote_file
        .split('/')
        .fold(destination_dir.join("files"), |path, segment| {
            path.join(segment)
        });
    download_hf_file(
        HuggingFaceRepoKind::Dataset,
        repo,
        revision,
        remote_file,
        &destination,
        true,
    )?;
    Ok(destination)
}

fn discover_hf_dataset_files(
    repo: &str,
    revision: &str,
    config: Option<&str>,
    split: &str,
) -> Result<Vec<String>, String> {
    validate_hf_repo_id(repo)?;
    if split.trim().is_empty() {
        return Err("Hugging Face dataset split must not be empty".to_string());
    }
    let url = format!(
        "https://huggingface.co/api/datasets/{}/revision/{}",
        hf_encode_path(repo),
        hf_encode_component(revision)
    );
    let metadata = hf_get_json(&url, &format!("dataset metadata for {repo}@{revision}"))?;
    let siblings = metadata
        .get("siblings")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            format!(
                "Hugging Face dataset metadata for {repo}@{revision} is missing a siblings file list"
            )
        })?;
    let mut candidates = siblings
        .iter()
        .filter_map(|entry| entry.get("rfilename").and_then(Value::as_str))
        .filter(|path| is_hf_native_dataset_path(path))
        .map(str::to_string)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(format!(
            "Hugging Face dataset {repo}@{revision} exposes no JSONL/NDJSON/Parquet files; the native backend does not execute dataset builders or silently convert unsupported formats such as CSV"
        ));
    }

    if let Some(config) = config.filter(|value| !value.trim().is_empty()) {
        let matches = candidates
            .iter()
            .filter(|path| hf_path_has_component(path, config))
            .cloned()
            .collect::<Vec<_>>();
        if !matches.is_empty() {
            candidates = matches;
        }
    }

    let split_matches = candidates
        .iter()
        .filter(|path| hf_path_matches_split(path, split))
        .cloned()
        .collect::<Vec<_>>();
    if !split_matches.is_empty() {
        if split_matches.len() > 1
            && !split_matches
                .iter()
                .all(|path| hf_path_looks_like_ordered_shard(path))
        {
            return Err(format!(
                "Hugging Face dataset {repo}@{revision} has multiple non-sharded JSONL/NDJSON/Parquet files matching split {split:?}; pass --hf-dataset-file PATH or --hf_dataset_config to disambiguate"
            ));
        }
        candidates = split_matches;
    } else if candidates.len() != 1 {
        return Err(format!(
            "Hugging Face dataset {repo}@{revision} has {} JSONL/NDJSON/Parquet files but none can be selected unambiguously for split {split:?}; pass --hf-dataset-file PATH",
            candidates.len()
        ));
    }
    candidates.sort();
    candidates.dedup();
    Ok(candidates)
}

fn fetch_hf_dataset_files(
    repo: &str,
    revision: &str,
    remote_files: &[String],
    cache_root: &Path,
    config: Option<&str>,
    split: &str,
) -> Result<PathBuf, String> {
    if remote_files.is_empty() {
        return Err("Hugging Face dataset selection produced no files".to_string());
    }
    let mut local_files = Vec::with_capacity(remote_files.len());
    for remote_file in remote_files {
        let local = fetch_hf_dataset_file(repo, revision, remote_file, cache_root)?;
        local_files.push(if is_hf_parquet_path(remote_file) {
            materialize_parquet_jsonl(&local)?
        } else {
            local
        });
    }
    if local_files.len() == 1 {
        return Ok(local_files.remove(0));
    }

    let destination_dir =
        hf_repo_cache_dir(cache_root, HuggingFaceRepoKind::Dataset, repo, revision)?;
    let config_label = config
        .filter(|value| !value.trim().is_empty())
        .map(hf_cache_slug)
        .unwrap_or_else(|| "default".to_string());
    let destination = destination_dir.join(format!(
        "combined-{config_label}-{}.jsonl",
        hf_cache_slug(split)
    ));
    if destination.is_file() {
        return Ok(destination);
    }
    fs::create_dir_all(&destination_dir)
        .map_err(|error| format!("could not create {}: {error}", destination_dir.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock cannot create dataset staging name: {error}"))?
        .as_nanos();
    let temp = destination_dir.join(format!(
        ".combined-{}-{}-{nonce}.part",
        std::process::id(),
        hf_cache_slug(split)
    ));
    let combine_result = (|| -> Result<(), String> {
        let mut output = BufWriter::new(
            File::create(&temp)
                .map_err(|error| format!("could not create {}: {error}", temp.display()))?,
        );
        for local in &local_files {
            let mut input = BufReader::new(
                File::open(local)
                    .map_err(|error| format!("could not open {}: {error}", local.display()))?,
            );
            io::copy(&mut input, &mut output)
                .map_err(|error| format!("could not combine {}: {error}", local.display()))?;
            output
                .write_all(b"\n")
                .map_err(|error| format!("could not delimit {}: {error}", local.display()))?;
        }
        output
            .flush()
            .map_err(|error| format!("could not flush {}: {error}", temp.display()))?;
        output
            .get_ref()
            .sync_all()
            .map_err(|error| format!("could not sync {}: {error}", temp.display()))?;
        drop(output);
        fs::rename(&temp, &destination).map_err(|error| {
            format!(
                "could not publish combined Hugging Face dataset {} -> {}: {error}",
                temp.display(),
                destination.display()
            )
        })?;
        Ok(())
    })();
    if let Err(error) = combine_result {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    Ok(destination)
}

fn hf_get_json(url: &str, label: &str) -> Result<Value, String> {
    let mut request = ureq::get(url).set("User-Agent", "hierarchos-native-cli/0.1");
    if let Some(token) = hf_auth_token() {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(status, _)) => {
            return Err(format!(
                "Hugging Face returned HTTP {status} while reading {label}"
            ))
        }
        Err(ureq::Error::Transport(error)) => {
            return Err(format!(
                "Hugging Face transport failed while reading {label}: {error}"
            ))
        }
    };
    let mut body = String::new();
    response
        .into_reader()
        .read_to_string(&mut body)
        .map_err(|error| format!("could not read Hugging Face response for {label}: {error}"))?;
    serde_json::from_str(&body)
        .map_err(|error| format!("Hugging Face returned invalid JSON for {label}: {error}"))
}

fn is_hf_jsonl_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".jsonl") || lower.ends_with(".ndjson")
}

fn is_hf_parquet_path(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".parquet")
}

fn is_hf_native_dataset_path(path: &str) -> bool {
    is_hf_jsonl_path(path) || is_hf_parquet_path(path)
}

fn materialize_parquet_jsonl(source: &Path) -> Result<PathBuf, String> {
    if !source.is_file() {
        return Err(format!(
            "Parquet dataset does not exist: {}",
            source.display()
        ));
    }
    let destination = source.with_extension("parquet.jsonl");
    if destination.is_file() {
        return Ok(destination);
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock cannot create Parquet staging name: {error}"))?
        .as_nanos();
    let temp = parent.join(format!(
        ".{}.native-parquet-{}-{nonce}.part",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("dataset.jsonl"),
        std::process::id()
    ));
    let convert_result = (|| -> Result<(), String> {
        let file = File::open(source)
            .map_err(|error| format!("could not open {}: {error}", source.display()))?;
        let reader = SerializedFileReader::new(file)
            .map_err(|error| format!("could not decode Parquet {}: {error}", source.display()))?;
        let rows = reader
            .get_row_iter(None)
            .map_err(|error| format!("could not iterate Parquet {}: {error}", source.display()))?;
        let mut output = BufWriter::new(
            File::create(&temp)
                .map_err(|error| format!("could not create {}: {error}", temp.display()))?,
        );
        for row in rows {
            let row: Row = row.map_err(|error| {
                format!(
                    "could not read Parquet row from {}: {error}",
                    source.display()
                )
            })?;
            serde_json::to_writer(&mut output, &row.to_json_value()).map_err(|error| {
                format!(
                    "could not encode Parquet row from {}: {error}",
                    source.display()
                )
            })?;
            output.write_all(b"\n").map_err(|error| {
                format!(
                    "could not delimit Parquet row from {}: {error}",
                    source.display()
                )
            })?;
        }
        output
            .flush()
            .map_err(|error| format!("could not flush {}: {error}", temp.display()))?;
        output
            .get_ref()
            .sync_all()
            .map_err(|error| format!("could not sync {}: {error}", temp.display()))?;
        drop(output);
        fs::rename(&temp, &destination).map_err(|error| {
            format!(
                "could not publish native Parquet conversion {} -> {}: {error}",
                temp.display(),
                destination.display()
            )
        })?;
        Ok(())
    })();
    if let Err(error) = convert_result {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    Ok(destination)
}

fn hf_path_has_component(path: &str, component: &str) -> bool {
    path.split('/')
        .any(|part| part.eq_ignore_ascii_case(component))
}

fn hf_path_matches_split(path: &str, split: &str) -> bool {
    let split = split.to_ascii_lowercase();
    let filename = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    let stem = filename
        .strip_suffix(".jsonl")
        .or_else(|| filename.strip_suffix(".ndjson"))
        .or_else(|| filename.strip_suffix(".parquet"))
        .unwrap_or(&filename);
    stem == split
        || stem.starts_with(&format!("{split}-"))
        || stem.starts_with(&format!("{split}_"))
        || hf_path_has_component(path, &split)
}

fn hf_path_looks_like_ordered_shard(path: &str) -> bool {
    let filename = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    filename.contains("-of-")
        && filename.split("-of-").next().is_some_and(|prefix| {
            prefix
                .rsplit('-')
                .next()
                .is_some_and(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
        })
}

fn hf_repo_cache_dir(
    cache_root: &Path,
    kind: HuggingFaceRepoKind,
    repo: &str,
    revision: &str,
) -> Result<PathBuf, String> {
    validate_hf_repo_id(repo)?;
    if revision.trim().is_empty() {
        return Err("Hugging Face revision must not be empty".to_string());
    }
    Ok(cache_root
        .join(kind.cache_label())
        .join(hf_cache_slug(repo))
        .join(hf_cache_slug(revision)))
}

fn download_hf_file(
    kind: HuggingFaceRepoKind,
    repo: &str,
    revision: &str,
    remote_file: &str,
    destination: &Path,
    required: bool,
) -> Result<bool, String> {
    validate_hf_relative_path(remote_file)?;
    if destination.is_file() {
        return Ok(true);
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let url = format!(
        "https://huggingface.co/{}{}/resolve/{}/{}?download=true",
        kind.url_prefix(),
        hf_encode_path(repo),
        hf_encode_component(revision),
        hf_encode_path(remote_file)
    );
    let mut request = ureq::get(&url).set("User-Agent", "hierarchos-native-cli/0.1");
    if let Some(token) = hf_auth_token() {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(404, _)) if !required => return Ok(false),
        Err(ureq::Error::Status(status, _)) => {
            return Err(format!(
                "Hugging Face returned HTTP {status} for {repo}@{revision}:{remote_file}"
            ))
        }
        Err(ureq::Error::Transport(error)) => {
            return Err(format!(
                "Hugging Face transport failed for {repo}@{revision}:{remote_file}: {error}"
            ))
        }
    };
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock cannot create download staging name: {error}"))?
        .as_nanos();
    let temp = parent.join(format!(
        ".{}.part-{}-{nonce}",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("hf-download"),
        std::process::id()
    ));
    let write_result = (|| -> Result<(), String> {
        let mut reader = response.into_reader();
        let mut file = File::create(&temp)
            .map_err(|error| format!("could not create {}: {error}", temp.display()))?;
        io::copy(&mut reader, &mut file)
            .map_err(|error| format!("could not download {url}: {error}"))?;
        file.flush()
            .map_err(|error| format!("could not flush {}: {error}", temp.display()))?;
        file.sync_all()
            .map_err(|error| format!("could not sync {}: {error}", temp.display()))?;
        fs::rename(&temp, destination).map_err(|error| {
            format!(
                "could not publish downloaded Hugging Face file {} -> {}: {error}",
                temp.display(),
                destination.display()
            )
        })?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    Ok(true)
}

fn hf_auth_token() -> Option<String> {
    env::var("HF_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("HUGGING_FACE_HUB_TOKEN")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
}

fn validate_hf_repo_id(repo: &str) -> Result<(), String> {
    let mut parts = repo.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty()
        || name.is_empty()
        || parts.next().is_some()
        || owner == "."
        || owner == ".."
        || name == "."
        || name == ".."
        || repo.contains('\\')
    {
        return Err(format!(
            "invalid Hugging Face repo id {repo:?}; expected OWNER/REPO"
        ));
    }
    Ok(())
}

fn validate_hf_relative_path(path: &str) -> Result<(), String> {
    if path.trim().is_empty() || path.starts_with('/') || path.starts_with('\\') {
        return Err(format!("invalid Hugging Face relative file path {path:?}"));
    }
    if path
        .replace('\\', "/")
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!("invalid Hugging Face relative file path {path:?}"));
    }
    Ok(())
}

fn hf_cache_slug(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn hf_encode_path(value: &str) -> String {
    value
        .split('/')
        .map(hf_encode_component)
        .collect::<Vec<_>>()
        .join("/")
}

fn hf_encode_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for &byte in value.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            let _ = write!(&mut out, "%{byte:02X}");
        }
    }
    out
}

fn prepare_training_dataset(
    source: &Path,
    model_dir: &Path,
    tokenizer_override: Option<&Path>,
    output_dir: &Path,
    options: &TextDatasetOptions,
) -> Result<PathBuf, String> {
    if let Some(path) = tokenizer_override {
        let tokenizer_path = resolve_local_tokenizer_path(path)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
        validate_tokenizer_vocab_for_package(&tokenizer, model_dir, &tokenizer_path)?;
    }
    if source.is_dir() {
        return Ok(source.to_path_buf());
    }
    if !source.is_file() {
        return Err(format!(
            "training dataset does not exist: {}",
            source.display()
        ));
    }
    let input = File::open(source).map_err(|error| {
        format!(
            "could not open training dataset {}: {error}",
            source.display()
        )
    })?;
    let mut first_object = None;
    for line in BufReader::new(input).lines() {
        let line = line.map_err(|error| format!("could not read {}: {error}", source.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        first_object = Some(
            serde_json::from_str::<Value>(&line)
                .map_err(|error| format!("{} is not JSONL: {error}", source.display()))?,
        );
        break;
    }
    let first =
        first_object.ok_or_else(|| format!("training dataset is empty: {}", source.display()))?;
    if first.get("input_ids").and_then(Value::as_array).is_some() {
        return Ok(source.to_path_buf());
    }

    let tokenizer_path = match tokenizer_override {
        Some(path) => resolve_local_tokenizer_path(path)?,
        None => model_dir.join("tokenizer.json"),
    };
    if !tokenizer_path.is_file() {
        return Err(format!(
            "raw-text native training requires {}; use a schema-v6 token cache or tokenized JSONL otherwise",
            tokenizer_path.display()
        ));
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
    validate_tokenizer_vocab_for_package(&tokenizer, model_dir, &tokenizer_path)?;
    let eos_token_id = resolve_eos_token_id(&tokenizer, &tokenizer_path)?;
    let output_parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent)
        .map_err(|error| format!("could not create {}: {error}", output_parent.display()))?;
    let output_name = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hierarchos-model");
    let tokenized_path = output_parent.join(format!(".{output_name}.native-tokenized.jsonl"));
    tokenize_jsonl(source, &tokenized_path, &tokenizer, eos_token_id, options)?;
    eprintln!(
        "native dataset: tokenized {} -> {}",
        source.display(),
        tokenized_path.display()
    );
    Ok(tokenized_path)
}

fn resolve_local_tokenizer_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_dir() {
        let tokenizer = path.join("tokenizer.json");
        if tokenizer.is_file() {
            return Ok(tokenizer);
        }
        return Err(format!(
            "local tokenizer directory is missing {}",
            tokenizer.display()
        ));
    }
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    Err(format!(
        "local tokenizer path does not exist: {}",
        path.display()
    ))
}

fn package_vocab_size(model_dir: &Path) -> Result<usize, String> {
    for name in [
        "hierarchos_rust_config.json",
        "hierarchos_config.json",
        "config.json",
    ] {
        let path = model_dir.join(name);
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
        let vocab = value.get("vocab_size").and_then(Value::as_u64).or_else(|| {
            value
                .get("architecture_contract")
                .and_then(|contract| contract.get("vocab_size"))
                .and_then(Value::as_u64)
        });
        if let Some(vocab) = vocab {
            return usize::try_from(vocab)
                .map_err(|_| format!("{} vocab_size exceeds usize", path.display()));
        }
    }
    Err(format!(
        "model package {} does not declare vocab_size in hierarchos_rust_config.json or hierarchos_config.json",
        model_dir.display()
    ))
}

fn validate_tokenizer_vocab_for_package(
    tokenizer: &Tokenizer,
    model_dir: &Path,
    tokenizer_path: &Path,
) -> Result<(), String> {
    let expected = package_vocab_size(model_dir)?;
    let actual = tokenizer.get_vocab_size(true);
    if actual != expected {
        return Err(format!(
            "tokenizer/model vocabulary mismatch for {}: tokenizer={} model={}",
            tokenizer_path.display(),
            actual,
            expected
        ));
    }
    Ok(())
}

fn install_local_tokenizer_assets(source: &Path, destination_dir: &Path) -> Result<(), String> {
    let tokenizer_path = resolve_local_tokenizer_path(source)?;
    fs::create_dir_all(destination_dir).map_err(|error| {
        format!(
            "could not create tokenizer destination {}: {error}",
            destination_dir.display()
        )
    })?;
    fs::copy(&tokenizer_path, destination_dir.join("tokenizer.json")).map_err(|error| {
        format!(
            "could not copy local tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    let source_dir = tokenizer_path.parent().unwrap_or_else(|| Path::new("."));
    for name in TOKENIZER_ASSETS
        .iter()
        .copied()
        .filter(|name| *name != "tokenizer.json")
    {
        let asset = source_dir.join(name);
        if asset.is_file() {
            fs::copy(&asset, destination_dir.join(name))
                .map_err(|error| format!("could not copy {}: {error}", asset.display()))?;
        }
    }
    Ok(())
}

fn install_local_tokenizer_assets_into_checkpoints(
    source: &Path,
    output_dir: &Path,
) -> Result<(), String> {
    if !output_dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(output_dir)
        .map_err(|error| format!("could not inspect {}: {error}", output_dir.display()))?
    {
        let entry =
            entry.map_err(|error| format!("could not inspect checkpoint entry: {error}"))?;
        if entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
            && entry
                .file_name()
                .to_string_lossy()
                .starts_with("checkpoint-")
        {
            install_local_tokenizer_assets(source, &entry.path())?;
        }
    }
    Ok(())
}

fn special_token_content(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| {
        value
            .as_object()
            .and_then(|object| object.get("content"))
            .and_then(Value::as_str)
    })
}

fn resolve_eos_token_id(tokenizer: &Tokenizer, tokenizer_path: &Path) -> Result<u32, String> {
    let tokenizer_dir = tokenizer_path.parent().unwrap_or_else(|| Path::new("."));
    for name in [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ] {
        let sidecar = tokenizer_dir.join(name);
        if !sidecar.is_file() {
            continue;
        }
        let document: Value = serde_json::from_slice(
            &fs::read(&sidecar)
                .map_err(|error| format!("could not read {}: {error}", sidecar.display()))?,
        )
        .map_err(|error| format!("could not parse {}: {error}", sidecar.display()))?;
        if let Some(id) = document.get("eos_token_id").and_then(Value::as_u64) {
            if let Ok(id) = u32::try_from(id) {
                if usize::try_from(id)
                    .ok()
                    .is_some_and(|id| id < tokenizer.get_vocab_size(true))
                {
                    return Ok(id);
                }
            }
        }
        if let Some(token) = document.get("eos_token").and_then(special_token_content) {
            if let Some(id) = tokenizer.token_to_id(token) {
                return Ok(id);
            }
        }
    }
    for token in ["<|endoftext|>", "</s>", "<eos>", "<EOS>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            return Ok(id);
        }
    }
    Err(format!(
        "raw-text native training requires a resolvable EOS token for {}; add tokenizer_config.json/special_tokens_map.json with eos_token or use a tokenizer that exposes a standard EOS token",
        tokenizer_path.display()
    ))
}

fn tokenize_jsonl(
    source: &Path,
    destination: &Path,
    tokenizer: &Tokenizer,
    eos_token_id: u32,
    options: &TextDatasetOptions,
) -> Result<(), String> {
    let input = BufReader::new(
        File::open(source)
            .map_err(|error| format!("could not open {}: {error}", source.display()))?,
    );
    let mut output = BufWriter::new(
        File::create(destination)
            .map_err(|error| format!("could not create {}: {error}", destination.display()))?,
    );
    let mut written = 0usize;
    for (line_index, line) in input.lines().enumerate() {
        let line = line.map_err(|error| format!("could not read {}: {error}", source.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|error| format!("invalid JSON on line {}: {error}", line_index + 1))?;
        let Some((ids, labels, weights)) = tokenize_row(tokenizer, &row, eos_token_id, options)
            .map_err(|error| format!("line {}: {error}", line_index + 1))?
        else {
            continue;
        };
        if ids.len() < 2 {
            continue;
        }
        let record = json!({
            "input_ids": ids,
            "labels": labels,
            "attention_mask": vec![1.0f32; weights.len()],
            "loss_weights": weights,
        });
        serde_json::to_writer(&mut output, &record)
            .map_err(|error| format!("could not encode tokenized row: {error}"))?;
        output.write_all(b"\n").map_err(|error| error.to_string())?;
        written += 1;
    }
    output.flush().map_err(|error| error.to_string())?;
    if written == 0 {
        return Err("native tokenizer produced no trainable rows".to_string());
    }
    Ok(())
}

fn tokenize_row(
    tokenizer: &Tokenizer,
    row: &Value,
    eos_token_id: u32,
    options: &TextDatasetOptions,
) -> Result<Option<(Vec<u32>, Vec<i64>, Vec<f32>)>, String> {
    if !row.is_object() {
        return Ok(None);
    }
    let (text_column, prompt_column, completion_column) = resolve_text_sample_columns(row, options);

    if let Some(text_column) = text_column.as_deref() {
        let Some(text) = string_field(row, text_column) else {
            return Ok(None);
        };
        if text.trim().is_empty() {
            return Ok(None);
        }
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|error| error.to_string())?;
        let mut ids = encoding.get_ids().to_vec();
        ids.push(eos_token_id);
        if options.max_length > 0 && ids.len() > options.max_length {
            ids.truncate(options.max_length.saturating_sub(1));
            ids.push(eos_token_id);
        }
        let labels = ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>();
        let weights = vec![options.response_loss_weight; ids.len()];
        return Ok(Some((ids, labels, weights)));
    }

    let (Some(prompt_column), Some(completion_column)) =
        (prompt_column.as_deref(), completion_column.as_deref())
    else {
        return Ok(None);
    };
    let prompt = string_field(row, prompt_column).unwrap_or_default();
    let completion = string_field(row, completion_column).unwrap_or_default();

    let input = string_field(row, "input").unwrap_or_default();
    if options.drop_empty_completions && completion.trim().is_empty() {
        return Ok(None);
    }
    if prompt.trim().is_empty() && completion.trim().is_empty() && input.trim().is_empty() {
        return Ok(None);
    }

    let output_encoding = tokenizer
        .encode(completion.clone(), false)
        .map_err(|error| error.to_string())?;
    if !completion.trim().is_empty() && output_encoding.len() < options.min_response_tokens {
        return Ok(None);
    }

    let (prefix, response_ids) = if options.kayla {
        let feelings = string_field(row, "feelings").unwrap_or_default();
        let thought = string_field(row, "thought-process").unwrap_or_default();
        let mut prompt_text = format!("### Instruction:\n{}\n\n", prompt);
        if !feelings.is_empty() {
            prompt_text.push_str(&format!("### Feelings:\n{}\n\n", feelings));
        }
        let thought_text = format!("### Thought Process:\n{}\n\n", thought);
        let response_text = format!("### Response:\n{}", completion);
        let mut response = tokenizer
            .encode(thought_text, false)
            .map_err(|error| error.to_string())?
            .get_ids()
            .to_vec();
        response.extend_from_slice(
            tokenizer
                .encode(response_text, false)
                .map_err(|error| error.to_string())?
                .get_ids(),
        );
        (prompt_text, response)
    } else if options.alpaca
        || (prompt_column.eq_ignore_ascii_case("instruction")
            && completion_column.eq_ignore_ascii_case("output"))
    {
        let mut prefix = String::new();
        if !input.trim().is_empty() {
            prefix.push_str(&format!("### Previous Context:\n{}\n\n", input.trim()));
        }
        prefix.push_str(&format!("### Instruction:\n{}\n\n### Response:\n", prompt));
        (prefix, output_encoding.get_ids().to_vec())
    } else {
        let prefix = if input.trim().is_empty() {
            format!("User: {}\n\nAssistant: ", prompt)
        } else {
            format!("User: {}\n\nUser: {}\n\nAssistant: ", input.trim(), prompt)
        };
        (prefix, output_encoding.get_ids().to_vec())
    };
    let prompt_encoding = tokenizer
        .encode(prefix, true)
        .map_err(|error| error.to_string())?;
    let Some((ids, labels, weights, retained_response_tokens)) = compose_prompt_response(
        prompt_encoding.get_ids(),
        &response_ids,
        eos_token_id,
        options,
    ) else {
        return Ok(None);
    };
    if !completion.trim().is_empty() && retained_response_tokens < options.min_response_tokens {
        return Ok(None);
    }
    Ok(Some((ids, labels, weights)))
}

fn string_field(row: &Value, name: &str) -> Option<String> {
    match row.get(name)? {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn nonempty_field(row: &Value, name: &str) -> bool {
    string_field(row, name).is_some_and(|value| !value.trim().is_empty())
}

fn resolve_text_sample_columns(
    row: &Value,
    options: &TextDatasetOptions,
) -> (Option<String>, Option<String>, Option<String>) {
    if let Some(text_column) = options.text_column.as_ref() {
        return (Some(text_column.clone()), None, None);
    }
    if options.alpaca && options.prompt_column.is_none() && options.completion_column.is_none() {
        return (
            None,
            Some("instruction".to_string()),
            Some("output".to_string()),
        );
    }
    if options.prompt_column.is_some() || options.completion_column.is_some() {
        return (
            None,
            options.prompt_column.clone(),
            options.completion_column.clone(),
        );
    }
    for candidate in ["text", "content"] {
        if nonempty_field(row, candidate) {
            return (Some(candidate.to_string()), None, None);
        }
    }
    for (prompt, completion) in [
        ("instruction", "output"),
        ("prompt", "completion"),
        ("question", "answer"),
    ] {
        if nonempty_field(row, prompt) || nonempty_field(row, completion) {
            return (None, Some(prompt.to_string()), Some(completion.to_string()));
        }
    }
    (None, None, None)
}

fn compose_prompt_response(
    prompt_ids: &[u32],
    response_ids: &[u32],
    eos_token_id: u32,
    options: &TextDatasetOptions,
) -> Option<(Vec<u32>, Vec<i64>, Vec<f32>, usize)> {
    let mut prompt_ids = prompt_ids.to_vec();
    let mut full_response_ids = response_ids.to_vec();
    full_response_ids.push(eos_token_id);

    if options.max_length > 0
        && prompt_ids.len().saturating_add(full_response_ids.len()) > options.max_length
    {
        let min_answer_tokens = options.min_response_tokens.min(response_ids.len());
        let mut min_response_total = options.max_length.min(min_answer_tokens.saturating_add(1));
        if min_response_total == 0 {
            min_response_total = options.max_length.min(1);
        }
        let prompt_budget = options.max_length.saturating_sub(min_response_total);
        if prompt_ids.len() > prompt_budget {
            let keep_from = prompt_ids.len().saturating_sub(prompt_budget);
            prompt_ids.drain(..keep_from);
        }
        let response_budget = options.max_length.saturating_sub(prompt_ids.len());
        if response_budget == 0 {
            return None;
        }
        if full_response_ids.len() > response_budget {
            full_response_ids.truncate(response_budget);
            if let Some(last) = full_response_ids.last_mut() {
                *last = eos_token_id;
            }
        }
    }
    if prompt_ids.is_empty() && full_response_ids.is_empty() {
        return None;
    }

    let prompt_len = prompt_ids.len();
    let retained_response_tokens = full_response_ids.len().saturating_sub(1);
    let mut ids = prompt_ids;
    ids.extend_from_slice(&full_response_ids);
    let mut labels = ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>();
    if !options.train_prompt_tokens {
        for label in labels.iter_mut().take(prompt_len) {
            *label = -100;
        }
    }
    let prompt_weight = if options.train_prompt_tokens {
        options.prompt_loss_weight
    } else {
        0.0
    };
    let mut weights = vec![prompt_weight; prompt_len];
    let response_offset = weights.len();
    weights.extend(std::iter::repeat(options.response_loss_weight).take(full_response_ids.len()));
    let boundary = retained_response_tokens.min(options.response_boundary_tokens);
    for weight in weights.iter_mut().skip(response_offset).take(boundary) {
        *weight *= options.response_boundary_loss_weight;
    }
    Some((ids, labels, weights, retained_response_tokens))
}

#[derive(Clone, Debug)]
struct TransformerGenerateOptions {
    model_path: Option<PathBuf>,
    hf_model: Option<String>,
    hf_model_revision: String,
    hf_cache_dir: Option<PathBuf>,
    prompts: Vec<String>,
    prompt_file: Option<PathBuf>,
    generation_config: Option<PathBuf>,
    generation_overrides: BTreeMap<String, Value>,
    context_length: Option<usize>,
    device_index: Option<usize>,
    add_special_tokens: bool,
    skip_special_tokens: bool,
}

fn generation_override_integer(
    overrides: &mut BTreeMap<String, Value>,
    field: &str,
    raw: &str,
    option: &str,
) -> Result<(), String> {
    let value = raw
        .parse::<u64>()
        .map_err(|error| format!("invalid {option} value {raw:?}: {error}"))?;
    overrides.insert(field.to_owned(), Value::Number(value.into()));
    Ok(())
}

fn generation_override_number(
    overrides: &mut BTreeMap<String, Value>,
    field: &str,
    raw: &str,
    option: &str,
) -> Result<(), String> {
    let value = raw
        .parse::<f64>()
        .map_err(|error| format!("invalid {option} value {raw:?}: {error}"))?;
    if !value.is_finite() {
        return Err(format!("{option} must be finite"));
    }
    let number = serde_json::Number::from_f64(value)
        .ok_or_else(|| format!("{option} cannot be represented as JSON"))?;
    overrides.insert(field.to_owned(), Value::Number(number));
    Ok(())
}

fn generation_token_list_value(raw: &str, option: &str) -> Result<Value, String> {
    let raw = raw.trim();
    if raw.starts_with('[') {
        return serde_json::from_str::<Value>(raw)
            .map_err(|error| format!("invalid {option} JSON: {error}"));
    }
    let mut tokens = Vec::new();
    for item in raw
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let token = item
            .parse::<u64>()
            .map_err(|error| format!("invalid {option} token {item:?}: {error}"))?;
        tokens.push(Value::Number(token.into()));
    }
    if tokens.is_empty() {
        return Err(format!("{option} requires at least one token id"));
    }
    Ok(if tokens.len() == 1 {
        tokens.pop().expect("one token")
    } else {
        Value::Array(tokens)
    })
}

fn parse_transformer_generate_options(
    mut argv: VecDeque<OsString>,
) -> Result<TransformerGenerateOptions, String> {
    let mut model_path = None;
    let mut hf_model = None;
    let mut hf_model_revision = "main".to_owned();
    let mut hf_cache_dir = None;
    let mut prompts = Vec::new();
    let mut prompt_file = None;
    let mut generation_config = None;
    let mut generation_overrides = BTreeMap::new();
    let mut continuous_batching_config = None::<serde_json::Map<String, Value>>;
    let mut context_length = None;
    let mut device_index = None;
    let mut add_special_tokens = true;
    let mut skip_special_tokens = true;

    while let Some(arg) = argv.pop_front() {
        let key = arg.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                println!(
                    "hierarchos-native-cli transformer-generate (--model-path MODEL | --hf-model OWNER/REPO) (--prompt TEXT [--prompt TEXT ...] | --prompt-file FILE) [options]\n\n\
Generation config precedence: model config.json < package generation_config.json < --generation-config FILE < CLI overrides.\n\
Common overrides: --max-new-tokens N --max-length N --min-new-tokens N --min-length N --do-sample/--no-do-sample --temperature F --top-k N --top-p F --top-h F --min-p F --typical-p F --epsilon-cutoff F --eta-cutoff F --repetition-penalty F --sequence-bias JSON --exponential-decay-length-penalty JSON --num-beams N --num-return-sequences N --length-penalty F --early-stopping/--no-early-stopping --no-repeat-ngram-size N --bad-words-ids JSON --suppress-tokens IDS --begin-suppress-tokens IDS --forced-bos-token-id N --forced-eos-token-id IDS --eos-token-id IDS --seed N.\n\
Generation execution: --use-cache/--no-use-cache --cache-implementation NAME (for example dynamic, static, or paged) --continuous-batching [--continuous-batch-max-requests N] --renormalize-logits/--no-renormalize-logits --remove-invalid-values/--no-remove-invalid-values --early-stopping-never. Repeating --prompt creates a request batch; continuous batching is opt-in and implies the native paged KV cache.\n\
Runtime options: --context-length N --device-index N --no-add-special-tokens --keep-special-tokens --hf-model-revision REV --hf-cache-dir DIR."
                );
                return Err("__help__".to_owned());
            }
            "--model-path" | "--model" => {
                model_path = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--hf-model" => hf_model = Some(required_string(&mut argv, &key)?),
            "--hf-model-revision" => hf_model_revision = required_string(&mut argv, &key)?,
            "--hf-cache-dir" => hf_cache_dir = Some(PathBuf::from(required(&mut argv, &key)?)),
            "--prompt" => prompts.push(required_string(&mut argv, &key)?),
            "--prompt-file" => prompt_file = Some(PathBuf::from(required(&mut argv, &key)?)),
            "--generation-config" => {
                generation_config = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--context-length" => context_length = Some(parse_required(&mut argv, &key)?),
            "--device-index" => device_index = Some(parse_required(&mut argv, &key)?),
            "--add-special-tokens" => add_special_tokens = true,
            "--no-add-special-tokens" => add_special_tokens = false,
            "--keep-special-tokens" => skip_special_tokens = false,
            "--skip-special-tokens" => skip_special_tokens = true,
            "--do-sample" => {
                generation_overrides.insert("do_sample".to_owned(), Value::Bool(true));
            }
            "--no-do-sample" => {
                generation_overrides.insert("do_sample".to_owned(), Value::Bool(false));
            }
            "--use-cache" => {
                generation_overrides.insert("use_cache".to_owned(), Value::Bool(true));
            }
            "--no-use-cache" => {
                generation_overrides.insert("use_cache".to_owned(), Value::Bool(false));
            }
            "--cache-implementation" => {
                generation_overrides.insert(
                    "cache_implementation".to_owned(),
                    Value::String(required_string(&mut argv, &key)?),
                );
            }
            "--continuous-batching" => {
                continuous_batching_config.get_or_insert_with(serde_json::Map::new);
            }
            "--continuous-batch-max-requests" => {
                let value = parse_required::<usize>(&mut argv, &key)?;
                if value == 0 {
                    return Err("--continuous-batch-max-requests must be >= 1".to_owned());
                }
                continuous_batching_config
                    .get_or_insert_with(serde_json::Map::new)
                    .insert(
                        "max_requests_per_batch".to_owned(),
                        Value::Number(value.into()),
                    );
            }
            "--renormalize-logits" => {
                generation_overrides.insert("renormalize_logits".to_owned(), Value::Bool(true));
            }
            "--no-renormalize-logits" => {
                generation_overrides.insert("renormalize_logits".to_owned(), Value::Bool(false));
            }
            "--remove-invalid-values" => {
                generation_overrides.insert("remove_invalid_values".to_owned(), Value::Bool(true));
            }
            "--no-remove-invalid-values" => {
                generation_overrides.insert("remove_invalid_values".to_owned(), Value::Bool(false));
            }
            "--early-stopping" => {
                generation_overrides.insert("early_stopping".to_owned(), Value::Bool(true));
            }
            "--no-early-stopping" => {
                generation_overrides.insert("early_stopping".to_owned(), Value::Bool(false));
            }
            "--early-stopping-never" => {
                generation_overrides.insert(
                    "early_stopping".to_owned(),
                    Value::String("never".to_owned()),
                );
            }
            "--max-new-tokens"
            | "--max-length"
            | "--min-new-tokens"
            | "--min-length"
            | "--top-k"
            | "--num-beams"
            | "--num-return-sequences"
            | "--no-repeat-ngram-size"
            | "--seed" => {
                let field = key.trim_start_matches("--").replace('-', "_");
                let raw = required_string(&mut argv, &key)?;
                generation_override_integer(&mut generation_overrides, &field, &raw, &key)?;
            }
            "--temperature"
            | "--top-p"
            | "--top-h"
            | "--min-p"
            | "--typical-p"
            | "--epsilon-cutoff"
            | "--eta-cutoff"
            | "--repetition-penalty"
            | "--length-penalty" => {
                let field = key.trim_start_matches("--").replace('-', "_");
                let raw = required_string(&mut argv, &key)?;
                generation_override_number(&mut generation_overrides, &field, &raw, &key)?;
            }
            "--forced-bos-token-id" => {
                let raw = required_string(&mut argv, &key)?;
                generation_override_integer(
                    &mut generation_overrides,
                    "forced_bos_token_id",
                    &raw,
                    &key,
                )?;
            }
            "--forced-eos-token-id" | "--eos-token-id" => {
                let field = key.trim_start_matches("--").replace('-', "_");
                let raw = required_string(&mut argv, &key)?;
                generation_overrides.insert(field, generation_token_list_value(&raw, &key)?);
            }
            "--suppress-tokens" | "--begin-suppress-tokens" => {
                let field = key.trim_start_matches("--").replace('-', "_");
                let raw = required_string(&mut argv, &key)?;
                let value = generation_token_list_value(&raw, &key)?;
                generation_overrides.insert(
                    field,
                    match value {
                        Value::Array(values) => Value::Array(values),
                        value => Value::Array(vec![value]),
                    },
                );
            }
            "--bad-words-ids" => {
                let raw = required_string(&mut argv, &key)?;
                let value: Value = serde_json::from_str(&raw)
                    .map_err(|error| format!("invalid --bad-words-ids JSON: {error}"))?;
                generation_overrides.insert("bad_words_ids".to_owned(), value);
            }
            "--sequence-bias" => {
                let raw = required_string(&mut argv, &key)?;
                let value: Value = serde_json::from_str(&raw)
                    .map_err(|error| format!("invalid --sequence-bias JSON: {error}"))?;
                generation_overrides.insert("sequence_bias".to_owned(), value);
            }
            "--exponential-decay-length-penalty" => {
                let raw = required_string(&mut argv, &key)?;
                let value: Value = serde_json::from_str(&raw).map_err(|error| {
                    format!("invalid --exponential-decay-length-penalty JSON: {error}")
                })?;
                generation_overrides.insert("exponential_decay_length_penalty".to_owned(), value);
            }
            other => return Err(format!("unknown transformer-generate option {other:?}")),
        }
    }

    if model_path.is_some() == hf_model.is_some() {
        return Err(
            "transformer-generate requires exactly one of --model-path or --hf-model".to_owned(),
        );
    }
    if prompts.is_empty() == prompt_file.is_none() {
        return Err(
            "transformer-generate requires either one or more --prompt values or exactly one --prompt-file"
                .to_owned(),
        );
    }
    if let Some(continuous) = continuous_batching_config {
        if generation_overrides.get("use_cache") == Some(&Value::Bool(false)) {
            return Err("--continuous-batching conflicts with --no-use-cache".to_owned());
        }
        if let Some(cache) = generation_overrides
            .get("cache_implementation")
            .and_then(Value::as_str)
        {
            if cache != "paged" {
                return Err(format!(
                    "--continuous-batching requires --cache-implementation paged, not {cache:?}"
                ));
            }
        }
        generation_overrides.insert("use_cache".to_owned(), Value::Bool(true));
        generation_overrides.insert(
            "cache_implementation".to_owned(),
            Value::String("paged".to_owned()),
        );
        generation_overrides.insert(
            "continuous_batching_config".to_owned(),
            Value::Object(continuous),
        );
    }
    if context_length == Some(0) {
        return Err("--context-length must be positive".to_owned());
    }
    Ok(TransformerGenerateOptions {
        model_path,
        hf_model,
        hf_model_revision,
        hf_cache_dir,
        prompts,
        prompt_file,
        generation_config,
        generation_overrides,
        context_length,
        device_index,
        add_special_tokens,
        skip_special_tokens,
    })
}

fn merge_transformer_generation_json(
    target: &mut serde_json::Map<String, Value>,
    path: &Path,
) -> Result<(), String> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("could not read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("could not decode {}: {error}", path.display()))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("{} must contain a top-level JSON object", path.display()))?;
    for (field, value) in object {
        target.insert(field.clone(), value.clone());
    }
    Ok(())
}

fn clear_package_paged_serving_opt_ins(target: &mut serde_json::Map<String, Value>) {
    if target.get("cache_implementation").and_then(Value::as_str) == Some("paged") {
        target.remove("cache_implementation");
    }
    target.remove("continuous_batching_config");
}

fn transformer_special_token_text(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| {
        value
            .as_object()
            .and_then(|object| object.get("content"))
            .and_then(Value::as_str)
    })
}

fn infer_transformer_special_token_id(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    kind: &str,
) -> Option<u32> {
    let field = format!("{kind}_token");
    for name in [
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let path = model_dir.join(name);
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        let Some(text) = value.get(&field).and_then(transformer_special_token_text) else {
            continue;
        };
        if let Some(id) = tokenizer.token_to_id(text) {
            return Some(id);
        }
    }
    None
}

fn load_transformer_generation_config(
    model_dir: &Path,
    explicit_generation_config: Option<&Path>,
    overrides: &BTreeMap<String, Value>,
    tokenizer: &Tokenizer,
) -> Result<VulkanTransformerGenerationConfig, String> {
    let config_path = model_dir.join("config.json");
    let config_value: Value = serde_json::from_slice(
        &fs::read(&config_path)
            .map_err(|error| format!("could not read {}: {error}", config_path.display()))?,
    )
    .map_err(|error| format!("could not decode {}: {error}", config_path.display()))?;
    let mut merged = config_value.as_object().cloned().ok_or_else(|| {
        format!(
            "{} must contain a top-level JSON object",
            config_path.display()
        )
    })?;

    let package_generation_config = model_dir.join("generation_config.json");
    if package_generation_config.is_file() {
        merge_transformer_generation_json(&mut merged, &package_generation_config)?;
    }
    // Paged KV and the continuous scheduler are serving optimizations with
    // materially different allocation/scheduling behavior. Treat package
    // metadata as non-authoritative for these two policies so downloading a
    // model can never enable them implicitly. An explicit --generation-config
    // or CLI/GUI override below may still opt in deliberately.
    clear_package_paged_serving_opt_ins(&mut merged);
    if let Some(path) = explicit_generation_config {
        merge_transformer_generation_json(&mut merged, path)?;
    }
    for (field, value) in overrides {
        merged.insert(field.clone(), value.clone());
    }
    let mut config = VulkanTransformerGenerationConfig::from_hf_value(&Value::Object(merged))
        .map_err(|error| format!("invalid Transformer generation config: {error:#}"))?;
    if config.eos_token_ids.is_empty() {
        if let Some(eos) = infer_transformer_special_token_id(model_dir, tokenizer, "eos") {
            config.eos_token_ids.push(eos);
        }
    }
    config
        .validate()
        .map_err(|error| format!("invalid Transformer generation config: {error:#}"))?;
    Ok(config)
}

fn run_transformer_generate(argv: VecDeque<OsString>) -> Result<u8, String> {
    let options = match parse_transformer_generate_options(argv) {
        Ok(options) => options,
        Err(error) if error == "__help__" => return Ok(0),
        Err(error) => return Err(error),
    };
    let model_dir = if let Some(repo) = options.hf_model.as_deref() {
        let cache = resolve_hf_cache_root(options.hf_cache_dir.as_deref())?;
        fetch_hf_transformer_package(repo, &options.hf_model_revision, &cache, false)?
    } else {
        options
            .model_path
            .clone()
            .expect("validated local model path")
    };
    let package_config_path = model_dir.join("config.json");
    let package_config_value: Value =
        serde_json::from_slice(&fs::read(&package_config_path).map_err(|error| {
            format!("could not read {}: {error}", package_config_path.display())
        })?)
        .map_err(|error| format!("invalid {}: {error}", package_config_path.display()))?;
    let is_encoder_decoder = package_config_value
        .get("model_type")
        .and_then(Value::as_str)
        == Some("encoder-decoder");
    let (model_config, encoder_config) = if is_encoder_decoder {
        let encoder_value = package_config_value
            .get("encoder")
            .ok_or_else(|| "EncoderDecoderConfig is missing encoder sub-config".to_owned())?;
        let decoder_value = package_config_value
            .get("decoder")
            .ok_or_else(|| "EncoderDecoderConfig is missing decoder sub-config".to_owned())?;
        let encoder = VulkanTransformerConfig::from_hf_value(encoder_value).map_err(|error| {
            format!("invalid Hugging Face EncoderDecoder encoder config: {error:#}")
        })?;
        let decoder = VulkanTransformerConfig::from_hf_value(decoder_value).map_err(|error| {
            format!("invalid Hugging Face EncoderDecoder decoder config: {error:#}")
        })?;
        (decoder, Some(encoder))
    } else {
        (
            VulkanTransformerConfig::from_hf_value(&package_config_value)
                .map_err(|error| format!("invalid Hugging Face Transformer package: {error:#}"))?,
            None,
        )
    };
    if !model_config.uses_causal_attention() {
        return Err(format!(
            "transformer-generate requires a causal language model; {:?} is bidirectional",
            model_config.architecture
        ));
    }
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
    if tokenizer.get_vocab_size(true) > model_config.vocab_size {
        return Err(format!(
            "Transformer tokenizer vocabulary exceeds model embedding rows: tokenizer={} model={}",
            tokenizer.get_vocab_size(true),
            model_config.vocab_size
        ));
    }
    if let Some(encoder_config) = encoder_config.as_ref() {
        if tokenizer.get_vocab_size(true) > encoder_config.vocab_size {
            return Err(format!(
                "Transformer tokenizer vocabulary exceeds EncoderDecoder encoder embedding rows: tokenizer={} encoder={}",
                tokenizer.get_vocab_size(true),
                encoder_config.vocab_size
            ));
        }
    }
    let generation_config = load_transformer_generation_config(
        &model_dir,
        options.generation_config.as_deref(),
        &options.generation_overrides,
        &tokenizer,
    )?;
    let prompt_texts = if !options.prompts.is_empty() {
        options.prompts.clone()
    } else {
        let path = options
            .prompt_file
            .as_deref()
            .expect("validated prompt file");
        vec![fs::read_to_string(path)
            .map_err(|error| format!("could not read prompt file {}: {error}", path.display()))?]
    };
    if is_encoder_decoder && prompt_texts.len() != 1 {
        return Err(
            "batched --prompt generation is currently decoder-only; EncoderDecoder generation accepts one prompt"
                .to_owned(),
        );
    }
    let prompt_ids = prompt_texts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let encoding = tokenizer
                .encode(prompt.as_str(), options.add_special_tokens)
                .map_err(|error| {
                    format!("Transformer tokenizer encode failed for request {index}: {error}")
                })?;
            if encoding.get_ids().is_empty() {
                return Err(format!(
                    "Transformer tokenizer produced an empty prompt for request {index}"
                ));
            }
            Ok(encoding.get_ids().to_vec())
        })
        .collect::<Result<Vec<_>, String>>()?;
    if prompt_ids.len() > 1 && generation_config.continuous_batching_config.is_none() {
        eprintln!(
            "Transformer request batch will use compatibility serial scheduling; pass --continuous-batching to opt into the paged continuous scheduler"
        );
    }
    let source_max_position_embeddings = encoder_config
        .as_ref()
        .map(|config| config.max_position_embeddings)
        .unwrap_or(model_config.max_position_embeddings);
    for (index, prompt) in prompt_ids.iter().enumerate() {
        if prompt.len() > source_max_position_embeddings {
            return Err(format!(
                "request {index} has {} prompt tokens but model max_position_embeddings is {}",
                prompt.len(),
                source_max_position_embeddings
            ));
        }
    }
    let longest_prompt = prompt_ids.iter().map(Vec::len).max().unwrap_or(1);
    let requested_new_tokens = generation_config.max_new_tokens.unwrap_or_else(|| {
        let configured_max = generation_config
            .max_length
            .unwrap_or(model_config.max_position_embeddings);
        configured_max.saturating_sub(if is_encoder_decoder {
            1
        } else {
            longest_prompt
        })
    });
    let minimum_context = if is_encoder_decoder {
        1
    } else {
        longest_prompt
    };
    let desired_context = minimum_context
        .saturating_add(requested_new_tokens)
        .min(model_config.max_position_embeddings)
        .max(minimum_context);
    let context_length = options.context_length.unwrap_or(desired_context);
    if !is_encoder_decoder && context_length < longest_prompt {
        return Err(format!(
            "--context-length {context_length} is smaller than the longest encoded prompt length {longest_prompt}"
        ));
    }
    if context_length > model_config.max_position_embeddings {
        return Err(format!(
            "--context-length {context_length} exceeds model max_position_embeddings {}",
            model_config.max_position_embeddings
        ));
    }
    let generation_capacity = context_length.saturating_sub(minimum_context);
    if requested_new_tokens > generation_capacity {
        eprintln!(
            "Transformer generation length clipped by context: requested {requested_new_tokens} new token(s), capacity={} token(s)",
            generation_capacity
        );
    }

    let device = match options.device_index {
        Some(index) => VulkanDevice::new_with_index(index),
        None => VulkanDevice::new(),
    }
    .map_err(|error| format!("Vulkan device initialization failed: {error:#}"))?;
    let generated_batches = if is_encoder_decoder {
        let prompt_ids = &prompt_ids[0];
        let graph = VulkanSeq2SeqTransformer::from_hf_package(
            device,
            &model_dir,
            prompt_ids.len(),
            context_length,
        )
        .map_err(|error| format!("could not build Vulkan EncoderDecoder graph: {error:#}"))?;
        vec![graph
            .generate_sequences_ids(prompt_ids, None, &generation_config)
            .map_err(|error| format!("Vulkan EncoderDecoder generation failed: {error:#}"))?]
    } else {
        let graph = VulkanTransformer::from_hf_package(device, &model_dir, 1, context_length)
            .map_err(|error| format!("could not build Vulkan Transformer graph: {error:#}"))?;
        if prompt_ids.len() == 1 && generation_config.continuous_batching_config.is_none() {
            vec![graph
                .generate_sequences_ids(&prompt_ids[0], &generation_config)
                .map_err(|error| format!("Vulkan Transformer generation failed: {error:#}"))?]
        } else {
            graph
                .generate_batch_sequences_ids(&prompt_ids, &generation_config)
                .map_err(|error| format!("Vulkan Transformer batch generation failed: {error:#}"))?
        }
    };
    for (request_index, generated) in generated_batches.iter().enumerate() {
        if generated_batches.len() > 1 {
            println!("=== request {} ===", request_index + 1);
        }
        for (sequence_index, sequence) in generated.iter().enumerate() {
            let text = tokenizer
                .decode(sequence, options.skip_special_tokens)
                .map_err(|error| format!("Transformer tokenizer decode failed: {error}"))?;
            if generation_config.num_return_sequences > 1 {
                println!("--- sequence {} ---", sequence_index + 1);
            }
            println!("{text}");
        }
    }
    Ok(0)
}

#[derive(Clone, Debug)]
struct ChatOptions {
    model_path: PathBuf,
    prompt: Option<String>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    max_new_tokens: usize,
    entropy_stop_threshold: f32,
    entropy_stop_min_tokens: usize,
    entropy_stop_top_prob: f32,
    eos_stop_prob: f32,
    seed: u64,
    carry_chat_state: bool,
    raw_prompt: bool,
    chat_state_file: Option<PathBuf>,
    resume_chat_state_file: Option<PathBuf>,
}

fn parse_chat_options(mut argv: VecDeque<OsString>) -> Result<ChatOptions, String> {
    let mut model_path = None;
    let mut prompt = None;
    let mut temperature = 0.7f32;
    let mut top_k = 40usize;
    let mut top_p = 0.9f32;
    let mut repetition_penalty = 1.2f32;
    let mut max_new_tokens = 512usize;
    let mut entropy_stop_threshold = 0.0f32;
    let mut entropy_stop_min_tokens = 3usize;
    let mut entropy_stop_top_prob = 0.05f32;
    let mut eos_stop_prob = 0.0f32;
    let mut seed = 0u64;
    let mut carry_chat_state = false;
    let mut raw_prompt = false;
    let mut chat_state_file = None;
    let mut chat_state_auto = false;
    let mut resume_chat_state_file = None;
    while let Some(arg) = argv.pop_front() {
        let key = arg.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                println!("hierarchos-native-cli chat --model-path MODEL [--prompt TEXT] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--max-new-tokens N] [--entropy-stop-threshold F] [--entropy-stop-min-tokens N] [--entropy-stop-top-prob F] [--eos-stop-prob F] [--seed N] [--carry-chat-state] [--chat-state-file [PATH]] [--resume-chat-from-state-file PATH] [--raw-prompt]\n\nThe opt-in entropy/EOS stop guards use the same raw-logit softmax policy as the root CLI and are disabled by default. Chat state is saved as the backend-neutral Hierarchos runtime-state JSON interchange format. Resuming state implies --carry-chat-state.");
                return Err("__help__".to_string());
            }
            "--model-path" | "--model" => {
                model_path = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--prompt" => prompt = Some(required_string(&mut argv, &key)?),
            "--temperature" => temperature = parse_required(&mut argv, &key)?,
            "--top-k" => top_k = parse_required(&mut argv, &key)?,
            "--top-p" => top_p = parse_required(&mut argv, &key)?,
            "--repetition-penalty" => repetition_penalty = parse_required(&mut argv, &key)?,
            "--max-new-tokens" => max_new_tokens = parse_required(&mut argv, &key)?,
            "--entropy-stop-threshold" => entropy_stop_threshold = parse_required(&mut argv, &key)?,
            "--entropy-stop-min-tokens" => {
                entropy_stop_min_tokens = parse_required(&mut argv, &key)?
            }
            "--entropy-stop-top-prob" => entropy_stop_top_prob = parse_required(&mut argv, &key)?,
            "--eos-stop-prob" => eos_stop_prob = parse_required(&mut argv, &key)?,
            "--seed" => seed = parse_required(&mut argv, &key)?,
            "--carry-chat-state" => carry_chat_state = true,
            "--chat-state-file" => {
                if argv
                    .front()
                    .is_some_and(|next| !next.to_string_lossy().starts_with('-'))
                {
                    chat_state_file = Some(PathBuf::from(required(&mut argv, &key)?));
                } else {
                    chat_state_auto = true;
                }
            }
            "--resume-chat-from-state-file" => {
                resume_chat_state_file = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--raw-prompt" => raw_prompt = true,
            "--device" | "--threads" | "--chat-prefill-chunk-size" => {
                let _ = required(&mut argv, &key)?;
            }
            "--no-passive-learning" => {}
            other => return Err(format!("unsupported native chat argument {other:?}")),
        }
    }
    let model_path = model_path.ok_or("chat requires --model-path MODEL")?;
    if let Some(path) = resume_chat_state_file.as_ref() {
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pt"))
        {
            return Err(
                "native chat state uses the portable JSON runtime-state interchange format; legacy .pt chat state is intentionally unsupported"
                    .to_string(),
            );
        }
    }
    if resume_chat_state_file.is_some() {
        carry_chat_state = true;
    }
    for (name, value) in [
        ("--entropy-stop-threshold", entropy_stop_threshold),
        ("--entropy-stop-top-prob", entropy_stop_top_prob),
        ("--eos-stop-prob", eos_stop_prob),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and non-negative"));
        }
    }
    if entropy_stop_top_prob > 1.0 || eos_stop_prob > 1.0 {
        return Err("--entropy-stop-top-prob and --eos-stop-prob must not exceed 1.0".to_string());
    }
    if let Some(resume) = resume_chat_state_file.as_ref() {
        chat_state_file = Some(resume.clone());
    } else if chat_state_auto {
        chat_state_file = Some(default_chat_state_path(&model_path)?);
    }
    Ok(ChatOptions {
        model_path,
        prompt,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_new_tokens,
        entropy_stop_threshold,
        entropy_stop_min_tokens,
        entropy_stop_top_prob,
        eos_stop_prob,
        seed,
        carry_chat_state,
        raw_prompt,
        chat_state_file,
        resume_chat_state_file,
    })
}

fn default_chat_state_path(model_path: &Path) -> Result<PathBuf, String> {
    let model_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("model");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before UNIX_EPOCH: {error}"))?
        .as_secs();
    let directory = env::current_dir()
        .map_err(|error| format!("could not resolve current directory: {error}"))?
        .join("hierarchos_chat_states");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    Ok(directory.join(format!("{model_name}-{timestamp}.json")))
}

struct NativeSession {
    model: HierarchosModel,
    tokenizer: Tokenizer,
    state: RuntimeState,
    eos_id: Option<u32>,
}

impl NativeSession {
    fn load(model_dir: &Path) -> Result<Self, String> {
        let model = HierarchosModel::load(model_dir)
            .map_err(|error| format!("could not load native model: {error}"))?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?;
        if tokenizer.get_vocab_size(true) != model.config().vocab_size {
            return Err(format!(
                "tokenizer/model vocabulary mismatch: {} != {}",
                tokenizer.get_vocab_size(true),
                model.config().vocab_size
            ));
        }
        let eos_id = tokenizer.token_to_id("<|endoftext|>");
        let state = model.new_state();
        Ok(Self {
            model,
            tokenizer,
            state,
            eos_id,
        })
    }

    fn reset(&mut self) {
        self.state = self.model.new_state();
    }
}

fn run_chat(argv: VecDeque<OsString>) -> Result<u8, String> {
    let options = match parse_chat_options(argv) {
        Ok(options) => options,
        Err(error) if error == "__help__" => return Ok(0),
        Err(error) if error.starts_with("unsupported native chat argument") => {
            return Err(format!(
                "{error}; the pure-native chat runtime refuses non-native fallback"
            ));
        }
        Err(error) => return Err(error),
    };
    let mut session = NativeSession::load(&options.model_path)?;
    if let Some(path) = options.resume_chat_state_file.as_deref() {
        session.state = session
            .model
            .load_runtime_state_json(path)
            .map_err(|error| {
                format!(
                    "could not resume native chat state {}: {error}",
                    path.display()
                )
            })?;
        eprintln!("native chat state resumed: {}", path.display());
    }
    if let Some(prompt) = options.prompt.as_deref() {
        let text = generate_text(&mut session, prompt, &options)?;
        println!("{text}");
        persist_chat_state(&session, options.chat_state_file.as_deref())?;
        return Ok(0);
    }

    println!("Hierarchos native chat. /reset clears recurrent state; /status inspects state; /save persists state; /quit exits.");
    if let Some(path) = options.chat_state_file.as_deref() {
        println!("runtime state autosave: {}", path.display());
    }
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().map_err(|error| error.to_string())?;
        let mut line = String::new();
        if stdin
            .read_line(&mut line)
            .map_err(|error| error.to_string())?
            == 0
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }
        if line == "/reset" {
            session.reset();
            persist_chat_state(&session, options.chat_state_file.as_deref())?;
            println!("state reset");
            continue;
        }
        if line == "/status" {
            println!(
                "position={} history_tokens={} carry_state={} autosave={}",
                session.state.position(),
                session.state.history().len(),
                options.carry_chat_state,
                options
                    .chat_state_file
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "off".to_string())
            );
            continue;
        }
        if line == "/save" {
            let path = options.chat_state_file.as_deref().ok_or(
                "/save requires --chat-state-file [PATH] or --resume-chat-from-state-file PATH",
            )?;
            persist_chat_state(&session, Some(path))?;
            println!("state saved: {}", path.display());
            continue;
        }
        let text = generate_text(&mut session, line, &options)?;
        println!("{text}");
        persist_chat_state(&session, options.chat_state_file.as_deref())?;
    }
    persist_chat_state(&session, options.chat_state_file.as_deref())?;
    Ok(0)
}

fn persist_chat_state(session: &NativeSession, path: Option<&Path>) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    session
        .model
        .save_runtime_state_json(&session.state, path)
        .map_err(|error| {
            format!(
                "could not save native chat state {}: {error}",
                path.display()
            )
        })
}

fn should_stop_generation_from_uncertainty(
    logits: &[f32],
    generated_count: usize,
    eos_id: Option<u32>,
    entropy_threshold: f32,
    entropy_min_tokens: usize,
    top_prob_ceiling: f32,
    eos_prob_threshold: f32,
) -> bool {
    let entropy_guard_active = entropy_threshold > 0.0 && generated_count >= entropy_min_tokens;
    let eos_index = eos_id.and_then(|id| {
        let index = id as usize;
        (index < logits.len()).then_some(index)
    });
    let eos_guard_active = eos_prob_threshold > 0.0 && generated_count >= 1 && eos_index.is_some();
    if (!entropy_guard_active && !eos_guard_active) || logits.is_empty() {
        return false;
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return true;
    }

    let max_logit = logits.iter().copied().max_by(f32::total_cmp).unwrap_or(0.0);
    let mut exp_sum = 0.0f64;
    let mut weighted_log_sum = 0.0f64;
    let mut top_exp = 0.0f64;
    let mut eos_exp = 0.0f64;
    for (index, &logit) in logits.iter().enumerate() {
        let exp_value = f64::from(logit - max_logit).exp();
        if !exp_value.is_finite() {
            return true;
        }
        exp_sum += exp_value;
        top_exp = top_exp.max(exp_value);
        if eos_index == Some(index) {
            eos_exp = exp_value;
        }
    }
    if !exp_sum.is_finite() || exp_sum <= 0.0 {
        return true;
    }

    if eos_guard_active && eos_exp / exp_sum >= f64::from(eos_prob_threshold) {
        return true;
    }
    if !entropy_guard_active {
        return false;
    }

    let top_prob = top_exp / exp_sum;
    for &logit in logits {
        let probability = f64::from(logit - max_logit).exp() / exp_sum;
        weighted_log_sum += probability * (probability + 1.0e-10).ln();
    }
    let entropy = -weighted_log_sum;
    entropy >= f64::from(entropy_threshold)
        && (top_prob_ceiling <= 0.0 || top_prob <= f64::from(top_prob_ceiling))
}

fn generate_text(
    session: &mut NativeSession,
    prompt: &str,
    options: &ChatOptions,
) -> Result<String, String> {
    if !options.carry_chat_state {
        session.reset();
    }
    let formatted = if options.raw_prompt {
        prompt.to_string()
    } else {
        format!("User: {}\n\nAssistant: ", prompt.trim())
    };
    let encoding = session
        .tokenizer
        .encode(formatted, true)
        .map_err(|error| format!("tokenizer encode failed: {error}"))?;
    if encoding.is_empty() {
        return Err("tokenizer produced an empty prompt".to_string());
    }
    let mut logits = session
        .model
        .prefill_last(encoding.get_ids(), &mut session.state)
        .map_err(|error| format!("native prefill failed: {error}"))?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        repetition_penalty: options.repetition_penalty,
        seed: options.seed,
    });
    let mut response_ids = Vec::with_capacity(options.max_new_tokens);
    for _ in 0..options.max_new_tokens {
        if should_stop_generation_from_uncertainty(
            &logits,
            response_ids.len(),
            session.eos_id,
            options.entropy_stop_threshold,
            options.entropy_stop_min_tokens,
            options.entropy_stop_top_prob,
            options.eos_stop_prob,
        ) {
            break;
        }
        let token = sampler.sample(&logits, session.state.history());
        if session.eos_id.is_some_and(|eos| eos == token) {
            break;
        }
        response_ids.push(token);
        logits = session
            .model
            .step(token, &mut session.state)
            .map_err(|error| format!("native decode step failed: {error}"))?;
    }
    session
        .tokenizer
        .decode(&response_ids, false)
        .map_err(|error| format!("tokenizer decode failed: {error}"))
}

fn run_benchmark(mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let mut model_path = None;
    let mut prompt = "User: Explain why Vulkan compute is useful.\n\nAssistant: ".to_string();
    let mut iterations = 64usize;
    let mut external_requested = false;
    while let Some(arg) = argv.pop_front() {
        let key = arg.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                println!("hierarchos-native-cli benchmark --model-path MODEL [--prompt TEXT] [--benchmark-iterations N]\nThis pure-native command measures local Rust inference throughput. External benchmark catalogs are not dispatched to non-native runtimes.");
                return Ok(0);
            }
            "--model-path" | "--model" => {
                let value = required(&mut argv, &key)?;
                model_path = Some(PathBuf::from(&value));
            }
            "--prompt" => prompt = required_string(&mut argv, &key)?,
            "--benchmark-iterations" => iterations = parse_required(&mut argv, &key)?,
            "--benchmark" | "--benchmark-suite" | "--eval-tasks" | "--arc-agi-path" => {
                external_requested = true;
                let _ = required(&mut argv, &key)?;
            }
            "--benchmark-all"
            | "--benchmark-sequential"
            | "--list-benchmarks"
            | "--strict-benchmarks" => {
                external_requested = true;
            }
            _other => {
                // Preserve correctness by rejecting benchmark-catalog options the native local
                // throughput harness does not implement.
                external_requested = true;
                if let Some(next) = argv.front() {
                    if !next.to_string_lossy().starts_with('-') {
                        let _ = argv.pop_front();
                    }
                }
            }
        }
    }
    if external_requested {
        return Err(
            "external benchmark suites are not implemented by the pure-native benchmark runner; this binary will not dispatch to Python/PyTorch. Use the native local throughput benchmark or add a Rust benchmark adapter."
                .to_string(),
        );
    }
    let model_path = model_path.ok_or("benchmark requires --model-path MODEL")?;
    let mut session = NativeSession::load(&model_path)?;
    let encoding = session
        .tokenizer
        .encode(prompt, true)
        .map_err(|error| format!("tokenizer encode failed: {error}"))?;
    if encoding.is_empty() {
        return Err("benchmark prompt tokenized to zero tokens".to_string());
    }
    let prefill_started = Instant::now();
    let mut logits = session
        .model
        .prefill_last(encoding.get_ids(), &mut session.state)
        .map_err(|error| format!("prefill failed: {error}"))?;
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
    });
    let decode_started = Instant::now();
    for _ in 0..iterations {
        let token = sampler.sample(&logits, session.state.history());
        logits = session
            .model
            .step(token, &mut session.state)
            .map_err(|error| format!("decode failed: {error}"))?;
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend": "hierarchos-inference-rust-fp32-compute",
            "model": model_path,
            "prompt_tokens": encoding.len(),
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": encoding.len() as f64 / prefill_seconds.max(f64::MIN_POSITIVE),
            "decode_tokens": iterations,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": iterations as f64 / decode_seconds.max(f64::MIN_POSITIVE),
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(0)
}

fn run_devices(argv: VecDeque<OsString>) -> Result<u8, String> {
    if argv
        .front()
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        println!(
            "hierarchos-native-cli devices\nLists Vulkan compute devices visible to Hierarchos."
        );
        return Ok(0);
    }
    if !argv.is_empty() {
        return Err("devices takes no arguments".to_string());
    }
    let executable = find_vulkan_executable("hierarchos-vulkan-devices")
        .ok_or_else(|| "could not locate hierarchos-vulkan-devices".to_string())?;
    let status = Command::new(executable)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("could not launch Vulkan device probe: {error}"))?;
    Ok(status.code().unwrap_or(1).clamp(0, 255) as u8)
}

fn run_architectures(mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let mut json_output = false;
    while let Some(arg) = argv.pop_front() {
        match arg.to_string_lossy().as_ref() {
            "-h" | "--help" => {
                println!(
                    "hierarchos-native-cli architectures [--json]\n\n\
Lists the Hugging Face model_type contracts recognized by the native Vulkan Transformer backend.\n\
Canonical entries are native graph contracts. Package aliases are wrapper/config spellings that\n\
resolve to one of those native text graphs; they do not imply native vision/audio preprocessing."
                );
                return Ok(0);
            }
            "--json" => json_output = true,
            other => return Err(format!("unsupported architectures argument {other:?}")),
        }
    }

    let mut canonical = VulkanTransformerArchitecture::ALL
        .iter()
        .map(|architecture| architecture.model_type())
        .collect::<Vec<_>>();
    canonical.sort_unstable();
    canonical.dedup();
    let mut aliases = VulkanTransformerArchitecture::HF_MODEL_TYPE_ALIASES.to_vec();
    aliases.sort_unstable();
    aliases.dedup();

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "canonical_count": canonical.len(),
                "alias_count": aliases.len(),
                "advertised_model_type_count": canonical.len() + aliases.len(),
                "canonical_model_types": canonical,
                "package_aliases": aliases,
                "alias_scope": "text-backbone/config resolution; not full multimodal preprocessing",
            }))
            .map_err(|error| error.to_string())?
        );
        return Ok(0);
    }

    println!(
        "Native Vulkan Transformer architectures: {} canonical + {} package aliases\n",
        canonical.len(),
        aliases.len()
    );
    println!("Canonical model_type values (end-to-end native text graph):");
    for model_type in canonical {
        println!("  {model_type}");
    }
    println!("\nPackage aliases (native text-backbone/config resolution):");
    for model_type in aliases {
        println!("  {model_type}");
    }
    Ok(0)
}

fn run_doctor(mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let mut json_output = false;
    while let Some(arg) = argv.pop_front() {
        match arg.to_string_lossy().as_ref() {
            "-h" | "--help" => {
                println!(
                    "hierarchos-native-cli doctor [--json]\n\n\
Checks Vulkan loader/device visibility plus the companion binaries needed by Hierarchos training.\n\
Transformer training runs in this executable; coherent-v9 Hierarchos training uses the companion\n\
hierarchos-vulkan-train binary. HIERARCHOS_VULKAN_BIN_DIR may point to a separate binary directory."
                );
                return Ok(0);
            }
            "--json" => json_output = true,
            other => return Err(format!("unsupported doctor argument {other:?}")),
        }
    }

    let devices = VulkanDevice::enumerate_compute_devices()
        .map_err(|error| format!("Vulkan diagnostic failed: {error:#}"))?;
    let trainer = find_vulkan_executable("hierarchos-vulkan-train");
    let device_probe = find_vulkan_executable("hierarchos-vulkan-devices");
    let canonical_count = VulkanTransformerArchitecture::ALL.len();
    let alias_count = VulkanTransformerArchitecture::HF_MODEL_TYPE_ALIASES.len();

    if json_output {
        let device_values = devices
            .iter()
            .map(|device| {
                json!({
                    "index": device.index,
                    "name": device.name,
                    "device_type": device.device_type,
                    "compute_queue_family_index": device.compute_queue_family_index,
                    "device_uuid": device.device_uuid,
                    "driver_uuid": device.driver_uuid,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": !devices.is_empty(),
                "vulkan_compute_devices": device_values,
                "hierarchos_vulkan_train": trainer,
                "hierarchos_vulkan_devices": device_probe,
                "transformer_architectures": canonical_count,
                "transformer_package_aliases": alias_count,
            }))
            .map_err(|error| error.to_string())?
        );
    } else {
        println!("Hierarchos native backend doctor");
        println!("  Vulkan compute devices: {}", devices.len());
        for device in &devices {
            println!(
                "    [{}] {} ({}, queue family {})",
                device.index, device.name, device.device_type, device.compute_queue_family_index
            );
        }
        println!(
            "  hierarchos-vulkan-train: {}",
            trainer
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "not found (needed for Hierarchos train/finetune)".to_owned())
        );
        println!(
            "  hierarchos-vulkan-devices: {}",
            device_probe
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(
                    || "not found (optional; `doctor` probes Vulkan directly)".to_owned()
                )
        );
        println!("  Transformer registry: {canonical_count} canonical + {alias_count} aliases");
    }
    if devices.is_empty() {
        return Ok(2);
    }
    Ok(0)
}

fn run_merge_lora(mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let mut model_path: Option<PathBuf> = None;
    let mut adapter_path: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut overwrite = false;

    while let Some(arg) = argv.pop_front() {
        let key = arg.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                println!(
                    "hierarchos-native-cli merge-lora --model-path MODEL --lora-adapter-path ADAPTER --out-dir OUT [--overwrite-merge-output]\n\n\
Merges a bound Hierarchos PEFT-LoRA SafeTensors adapter into a standalone model package.\n\
The merge is pure Rust: no Python, PyTorch, PEFT runtime, CUDA runtime, or pickle loader is used.\n\
The adapter manifest must cryptographically bind the adapter to the exact base checkpoint and\n\
architecture contract. Standard LoRA, RS-LoRA scaling, rank/alpha patterns, fan-in/fan-out, and\n\
the project's optional saved LTM module are supported."
                );
                return Ok(0);
            }
            "--model-path" | "--model" => {
                model_path = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--lora-adapter-path" | "--adapter" => {
                adapter_path = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--out-dir" | "--output" => {
                output_dir = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--overwrite-merge-output" => overwrite = true,
            other => return Err(format!("unsupported merge-lora argument {other:?}")),
        }
    }

    let model_path = model_path.ok_or("merge-lora requires --model-path MODEL")?;
    let adapter_path = adapter_path.ok_or("merge-lora requires --lora-adapter-path ADAPTER")?;
    let output_dir = output_dir.ok_or("merge-lora requires --out-dir OUT")?;
    let source_dir = native_model_package_dir(&model_path)?;
    if !adapter_path.is_dir() {
        return Err(format!(
            "LoRA adapter directory does not exist: {}",
            adapter_path.display()
        ));
    }
    if output_dir.exists() && !overwrite {
        return Err(format!(
            "merge output already exists: {} (pass --overwrite-merge-output to replace it after validation)",
            output_dir.display()
        ));
    }
    if output_dir.exists() && !output_dir.is_dir() {
        return Err(format!(
            "merge output path exists but is not a directory: {}",
            output_dir.display()
        ));
    }

    let output_parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent).map_err(|error| {
        format!(
            "could not create merge output parent {}: {error}",
            output_parent.display()
        )
    })?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock cannot create merge staging name: {error}"))?
        .as_nanos();
    let output_name = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hierarchos-merged");
    let staging_dir = output_parent.join(format!(
        ".{output_name}.native-lora-staging-{}-{nonce}",
        std::process::id()
    ));
    if staging_dir.exists() {
        return Err(format!(
            "merge staging path unexpectedly exists: {}",
            staging_dir.display()
        ));
    }

    let build_result = (|| -> Result<_, String> {
        fs::create_dir(&staging_dir).map_err(|error| {
            format!(
                "could not create merge staging directory {}: {error}",
                staging_dir.display()
            )
        })?;
        copy_native_package_assets(&source_dir, &staging_dir, false)?;
        let base_weights = source_dir.join("model.safetensors");
        let merged_weights = staging_dir.join("model.safetensors");
        let report =
            merge_hierarchos_lora_safetensors(&base_weights, &adapter_path, &merged_weights)
                .map_err(|error| format!("native LoRA merge failed: {error:#}"))?;

        let provenance = json!({
            "format": "hierarchos-native-lora-merge-v1",
            "merge_runtime": "rust-native",
            "training_runtime": "vulkan",
            "base_checkpoint_sha256": report.base_checkpoint_sha256,
            "adapter_checkpoint_sha256": report.adapter_checkpoint_sha256,
            "architecture_contract_sha256": report.architecture_contract_sha256,
            "merged_lora_modules": report.merged_lora_modules,
            "replaced_module_tensors": report.replaced_module_tensors,
            "adapter_directory": adapter_path.file_name().map(|name| name.to_string_lossy().into_owned()),
        });
        let mut serialized = serde_json::to_vec_pretty(&provenance)
            .map_err(|error| format!("could not serialize native merge provenance: {error}"))?;
        serialized.push(b'\n');
        fs::write(staging_dir.join("hierarchos_lora_merge.json"), serialized)
            .map_err(|error| format!("could not write native merge provenance: {error}"))?;

        // This validates configuration, tokenizer, architecture, and every model
        // tensor expected by the native inference runtime before publication.
        let _ = NativeSession::load(&staging_dir)?;
        Ok(report)
    })();

    let report = match build_result {
        Ok(report) => report,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging_dir);
            return Err(error);
        }
    };

    publish_staged_directory(&staging_dir, &output_dir, overwrite)?;
    println!("native LoRA merge complete: {}", output_dir.display());
    println!(
        "merged_modules={} saved_module_tensors={} architecture_contract_sha256={}",
        report.merged_lora_modules,
        report.replaced_module_tensors,
        report.architecture_contract_sha256
    );
    Ok(0)
}

fn native_model_package_dir(input: &Path) -> Result<PathBuf, String> {
    if input.is_dir() {
        return Ok(input.to_path_buf());
    }
    if input.file_name().and_then(|name| name.to_str()) == Some("model.safetensors") {
        return Ok(input.parent().unwrap_or(Path::new(".")).to_path_buf());
    }
    Err(format!(
        "native model path must be a package directory or model.safetensors, got {}",
        input.display()
    ))
}

fn copy_native_package_assets(
    source_dir: &Path,
    destination_dir: &Path,
    include_model: bool,
) -> Result<(), String> {
    for name in ["hierarchos_rust_config.json", "hierarchos_config.json"] {
        let source = source_dir.join(name);
        if !source.is_file() {
            return Err(format!("source package is missing {}", source.display()));
        }
        fs::copy(&source, destination_dir.join(name))
            .map_err(|error| format!("could not copy {}: {error}", source.display()))?;
    }
    if include_model {
        let source = source_dir.join("model.safetensors");
        if !source.is_file() {
            return Err(format!("source package is missing {}", source.display()));
        }
        fs::copy(&source, destination_dir.join("model.safetensors"))
            .map_err(|error| format!("could not copy {}: {error}", source.display()))?;
    }
    for name in TOKENIZER_ASSETS {
        let source = source_dir.join(name);
        if source.is_file() {
            fs::copy(&source, destination_dir.join(name))
                .map_err(|error| format!("could not copy {}: {error}", source.display()))?;
        }
    }
    Ok(())
}

fn publish_staged_directory(staging: &Path, output: &Path, overwrite: bool) -> Result<(), String> {
    if !output.exists() {
        return fs::rename(staging, output).map_err(|error| {
            format!(
                "could not publish staged native package {} -> {}: {error}",
                staging.display(),
                output.display()
            )
        });
    }
    if !overwrite {
        return Err(format!("output already exists: {}", output.display()));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock cannot create backup name: {error}"))?
        .as_nanos();
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hierarchos-merged");
    let backup = parent.join(format!(
        ".{name}.native-lora-backup-{}-{nonce}",
        std::process::id()
    ));
    fs::rename(output, &backup).map_err(|error| {
        format!(
            "could not move existing merge output {} aside: {error}",
            output.display()
        )
    })?;
    if let Err(error) = fs::rename(staging, output) {
        let restore = fs::rename(&backup, output);
        return Err(match restore {
            Ok(()) => format!(
                "could not publish validated merge output; previous output was restored: {error}"
            ),
            Err(restore_error) => format!(
                "could not publish validated merge output ({error}) and could not restore previous output from {} ({restore_error})",
                backup.display()
            ),
        });
    }
    fs::remove_dir_all(&backup).map_err(|error| {
        format!(
            "new merge output was published, but old backup {} could not be removed: {error}",
            backup.display()
        )
    })?;
    Ok(())
}

fn run_ckpt_to_inference(mut argv: VecDeque<OsString>) -> Result<u8, String> {
    let mut input = None;
    let mut output = None;
    let mut tokenizer_path = None;
    while let Some(arg) = argv.pop_front() {
        let key = arg.to_string_lossy().into_owned();
        match key.as_str() {
            "-h" | "--help" => {
                println!("hierarchos-native-cli ckpt-2-inf --ckpt-input PACKAGE --inf-output OUT [--ckpt-tok-path LOCAL_TOKENIZER]\nNative SafeTensors/Vulkan packages are converted directly. A local tokenizer override is validated against model vocab_size and packaged atomically. Legacy .pt checkpoints are intentionally unsupported because this executable contains no Python/PyTorch bridge.");
                return Ok(0);
            }
            "--ckpt-input" | "--resume-from-ckpt" | "--model-path" => {
                input = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--inf-output" | "--out-dir" => {
                output = Some(PathBuf::from(required(&mut argv, &key)?))
            }
            "--ckpt-tok-path" => {
                tokenizer_path = Some(PathBuf::from(required(&mut argv, &key)?));
            }
            "--trust-remote-code" => {
                return Err(format!(
                    "{key} is unavailable in pure-native ckpt-2-inf; the native runtime never executes remote tokenizer code"
                ));
            }
            other => return Err(format!("unsupported ckpt-2-inf argument {other:?}")),
        }
    }
    let input =
        input.ok_or("ckpt-2-inf requires --ckpt-input, --resume-from-ckpt, or --model-path")?;
    if input
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pt"))
    {
        return Err(
            "legacy .pt checkpoints require a PyTorch object loader and are intentionally rejected by the pure-native runtime; convert once to the canonical SafeTensors package format before entering the native stack."
                .to_string(),
        );
    }
    let source_dir = native_model_package_dir(&input)?;
    let output = output.unwrap_or_else(|| source_dir.with_file_name("hierarchos_final"));
    if output.exists() {
        return Err(format!(
            "inference output already exists: {}",
            output.display()
        ));
    }

    let output_parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent).map_err(|error| {
        format!(
            "could not create inference output parent {}: {error}",
            output_parent.display()
        )
    })?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock cannot create inference staging name: {error}"))?
        .as_nanos();
    let output_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hierarchos-final");
    let staging_dir = output_parent.join(format!(
        ".{output_name}.native-inference-staging-{}-{nonce}",
        std::process::id()
    ));
    if staging_dir.exists() {
        return Err(format!(
            "inference staging path unexpectedly exists: {}",
            staging_dir.display()
        ));
    }

    let build_result = (|| -> Result<(), String> {
        fs::create_dir(&staging_dir).map_err(|error| {
            format!(
                "could not create inference staging directory {}: {error}",
                staging_dir.display()
            )
        })?;
        copy_native_package_assets(&source_dir, &staging_dir, true)?;
        if let Some(tokenizer_path) = tokenizer_path.as_deref() {
            let resolved = resolve_local_tokenizer_path(tokenizer_path)?;
            let tokenizer = Tokenizer::from_file(&resolved)
                .map_err(|error| format!("could not load {}: {error}", resolved.display()))?;
            validate_tokenizer_vocab_for_package(&tokenizer, &source_dir, &resolved)?;
            install_local_tokenizer_assets(tokenizer_path, &staging_dir)?;
        }
        // Validate the complete tokenizer/config/model package before exposing
        // it at the requested destination. A failed conversion must never
        // leave a directory that looks like a usable native inference model.
        let _ = NativeSession::load(&staging_dir)?;
        Ok(())
    })();
    if let Err(error) = build_result {
        let _ = fs::remove_dir_all(&staging_dir);
        return Err(error);
    }

    publish_staged_directory(&staging_dir, &output, false)?;
    println!("native inference package: {}", output.display());
    Ok(0)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn platform_executable(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn find_vulkan_executable(binary: &str) -> Option<PathBuf> {
    let name = platform_executable(binary);
    let mut candidates = Vec::new();
    if let Some(explicit_dir) = env::var_os("HIERARCHOS_VULKAN_BIN_DIR") {
        candidates.push(PathBuf::from(explicit_dir).join(&name));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join("vulkan").join(&name));
            candidates.push(exe_dir.join(&name));
        }
    }
    let root = repo_root();
    candidates.push(
        root.join("hierarchos-vulkan")
            .join("target")
            .join("release")
            .join(&name),
    );
    candidates.push(
        root.join("hierarchos-vulkan")
            .join("target")
            .join("debug")
            .join(&name),
    );
    candidates.into_iter().find(|path| path.is_file())
}

fn required(argv: &mut VecDeque<OsString>, option: &str) -> Result<OsString, String> {
    argv.pop_front()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn required_string(argv: &mut VecDeque<OsString>, option: &str) -> Result<String, String> {
    Ok(required(argv, option)?.to_string_lossy().into_owned())
}

fn parse_required<T>(argv: &mut VecDeque<OsString>, option: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = required_string(argv, option)?;
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {option} value {value:?}: {error}"))
}

fn forward_value(
    output: &mut Vec<OsString>,
    argv: &mut VecDeque<OsString>,
    canonical: &str,
    source: &str,
) -> Result<(), String> {
    output.push(OsString::from(canonical));
    output.push(required(argv, source)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_generate_cli_exposes_hf_generation_overrides() {
        let args = VecDeque::from(
            [
                "--model-path",
                "model",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "32",
                "--do-sample",
                "--temperature",
                "0.8",
                "--top-p",
                "0.91",
                "--num-beams",
                "3",
                "--num-return-sequences",
                "2",
                "--sequence-bias",
                "[[[4,5],-0.6],[[9],1.2]]",
                "--exponential-decay-length-penalty",
                "[5,1.1]",
                "--no-use-cache",
                "--cache-implementation",
                "paged",
                "--renormalize-logits",
                "--remove-invalid-values",
                "--early-stopping-never",
                "--eos-token-id",
                "2,7",
                "--suppress-tokens",
                "9,11",
                "--seed",
                "42",
                "--context-length",
                "128",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>(),
        );
        let options = parse_transformer_generate_options(args).expect("parse generation CLI");
        assert_eq!(options.model_path.as_deref(), Some(Path::new("model")));
        assert_eq!(options.prompts, vec!["hello"]);
        assert_eq!(options.context_length, Some(128));
        assert_eq!(options.generation_overrides["max_new_tokens"], json!(32));
        assert_eq!(options.generation_overrides["do_sample"], json!(true));
        assert_eq!(options.generation_overrides["temperature"], json!(0.8));
        assert_eq!(options.generation_overrides["top_p"], json!(0.91));
        assert_eq!(options.generation_overrides["num_beams"], json!(3));
        assert_eq!(
            options.generation_overrides["num_return_sequences"],
            json!(2)
        );
        assert_eq!(
            options.generation_overrides["sequence_bias"],
            json!([[[4, 5], -0.6], [[9], 1.2]])
        );
        assert_eq!(
            options.generation_overrides["exponential_decay_length_penalty"],
            json!([5, 1.1])
        );
        assert_eq!(options.generation_overrides["use_cache"], json!(false));
        assert_eq!(
            options.generation_overrides["cache_implementation"],
            json!("paged")
        );
        assert_eq!(
            options.generation_overrides["renormalize_logits"],
            json!(true)
        );
        assert_eq!(
            options.generation_overrides["remove_invalid_values"],
            json!(true)
        );
        assert_eq!(
            options.generation_overrides["early_stopping"],
            json!("never")
        );
        assert_eq!(options.generation_overrides["eos_token_id"], json!([2, 7]));
        assert_eq!(
            options.generation_overrides["suppress_tokens"],
            json!([9, 11])
        );
        assert_eq!(options.generation_overrides["seed"], json!(42));
    }

    #[test]
    fn transformer_generate_cli_continuous_batching_is_opt_in_and_implies_paged_cache() {
        let args = VecDeque::from(
            [
                "--model-path",
                "model",
                "--prompt",
                "first",
                "--prompt",
                "second",
                "--continuous-batching",
                "--continuous-batch-max-requests",
                "2",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>(),
        );
        let options =
            parse_transformer_generate_options(args).expect("parse continuous batching CLI");
        assert_eq!(options.prompts, vec!["first", "second"]);
        assert_eq!(options.generation_overrides["use_cache"], json!(true));
        assert_eq!(
            options.generation_overrides["cache_implementation"],
            json!("paged")
        );
        assert_eq!(
            options.generation_overrides["continuous_batching_config"],
            json!({"max_requests_per_batch": 2})
        );
    }

    #[test]
    fn transformer_generate_cli_continuous_batching_fails_closed_on_cache_conflicts() {
        for conflicting in [
            vec!["--continuous-batching", "--no-use-cache"],
            vec!["--continuous-batching", "--cache-implementation", "dynamic"],
        ] {
            let mut raw = vec!["--model-path", "model", "--prompt", "hello"];
            raw.extend(conflicting);
            let args = VecDeque::from(raw.into_iter().map(OsString::from).collect::<Vec<_>>());
            let error = parse_transformer_generate_options(args)
                .expect_err("continuous batching cache conflict must fail closed");
            assert!(
                error.contains("--continuous-batching"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn transformer_generation_config_overlay_is_last_write_wins() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hierarchos-transformer-generation-overlay-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create generation overlay fixture");
        let path = root.join("generation_config.json");
        fs::write(
            &path,
            br#"{"max_new_tokens":64,"temperature":0.7,"do_sample":true}"#,
        )
        .expect("write generation overlay fixture");
        let mut target = serde_json::Map::from_iter([
            ("max_new_tokens".to_owned(), json!(8)),
            ("top_p".to_owned(), json!(0.9)),
        ]);
        merge_transformer_generation_json(&mut target, &path).expect("merge generation config");
        assert_eq!(target["max_new_tokens"], json!(64));
        assert_eq!(target["temperature"], json!(0.7));
        assert_eq!(target["do_sample"], json!(true));
        assert_eq!(target["top_p"], json!(0.9));
        fs::remove_dir_all(&root).expect("remove generation overlay fixture");
    }

    #[test]
    fn transformer_package_metadata_cannot_implicitly_enable_paged_serving() {
        let mut package = serde_json::Map::from_iter([
            ("use_cache".to_owned(), json!(true)),
            ("cache_implementation".to_owned(), json!("paged")),
            (
                "continuous_batching_config".to_owned(),
                json!({"block_size": 16, "max_requests_per_batch": 4}),
            ),
            ("temperature".to_owned(), json!(0.7)),
        ]);

        clear_package_paged_serving_opt_ins(&mut package);

        assert_eq!(package["use_cache"], json!(true));
        assert_eq!(package["temperature"], json!(0.7));
        assert!(!package.contains_key("cache_implementation"));
        assert!(!package.contains_key("continuous_batching_config"));

        package.insert("cache_implementation".to_owned(), json!("static"));
        clear_package_paged_serving_opt_ins(&mut package);
        assert_eq!(package["cache_implementation"], json!("static"));
    }

    #[test]
    fn plain_text_row_produces_supervised_ids() {
        // The tokenizer itself is integration-tested by the native GUI/runtime. Keep this unit
        // test focused on JSON field resolution helpers that guard the native CLI layer.
        let row = json!({"text": "hello", "prompt": "ignored"});
        assert_eq!(string_field(&row, "text").as_deref(), Some("hello"));
    }

    #[test]
    fn platform_executable_matches_host() {
        let name = platform_executable("hierarchos-vulkan-train");
        if cfg!(windows) {
            assert!(name.ends_with(".exe"));
        } else {
            assert_eq!(name, "hierarchos-vulkan-train");
        }
    }

    #[test]
    fn native_cli_training_defaults_match_root_cli_contract() {
        let defaults = root_cli_training_defaults()
            .chunks_exact(2)
            .map(|pair| {
                (
                    pair[0].to_string_lossy().into_owned(),
                    pair[1].to_string_lossy().into_owned(),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(defaults.get("--epochs").map(String::as_str), Some("3"));
        assert_eq!(defaults.get("--batch-size").map(String::as_str), Some("64"));
        assert_eq!(defaults.get("--seed").map(String::as_str), Some("1337"));
        assert_eq!(defaults.get("--min-lr").map(String::as_str), Some("1e-6"));
        assert_eq!(
            defaults.get("--ponder-loss-weight").map(String::as_str),
            Some("0.01")
        );
    }

    #[test]
    fn native_uncertainty_stop_matches_root_cli_guard_geometry() {
        let uniform_logits = [0.0f32, 0.0, 0.0, 0.0];
        assert!(!should_stop_generation_from_uncertainty(
            &uniform_logits,
            2,
            Some(3),
            1.0,
            3,
            0.30,
            0.0,
        ));
        assert!(should_stop_generation_from_uncertainty(
            &uniform_logits,
            3,
            Some(3),
            1.0,
            3,
            0.30,
            0.0,
        ));
        assert!(!should_stop_generation_from_uncertainty(
            &uniform_logits,
            3,
            Some(3),
            1.0,
            3,
            0.20,
            0.0,
        ));
    }

    #[test]
    fn native_eos_probability_stop_uses_raw_softmax_and_generated_token_gate() {
        let logits = [0.0f32, 0.0, 2.0];
        assert!(!should_stop_generation_from_uncertainty(
            &logits,
            0,
            Some(2),
            0.0,
            3,
            0.05,
            0.5,
        ));
        assert!(should_stop_generation_from_uncertainty(
            &logits,
            1,
            Some(2),
            0.0,
            3,
            0.05,
            0.5,
        ));
    }

    #[test]
    fn native_uncertainty_stop_is_fail_closed_for_active_nonfinite_logits() {
        let logits = [0.0f32, f32::NAN];
        assert!(!should_stop_generation_from_uncertainty(
            &logits, 4, None, 0.0, 3, 0.05, 0.0,
        ));
        assert!(should_stop_generation_from_uncertainty(
            &logits, 4, None, 1.0, 3, 0.05, 0.0,
        ));
    }

    #[test]
    fn response_preserving_truncation_keeps_prompt_suffix_and_minimum_answer() {
        let options = TextDatasetOptions {
            max_length: 8,
            min_response_tokens: 3,
            ..TextDatasetOptions::default()
        };
        let prompt = [10, 11, 12, 13, 14, 15, 16];
        let response = [20, 21, 22, 23];
        let eos = 99;
        let (ids, labels, weights, retained_response_tokens) =
            compose_prompt_response(&prompt, &response, eos, &options)
                .expect("truncation should preserve a trainable row");

        assert_eq!(ids, vec![13, 14, 15, 16, 20, 21, 22, 99]);
        assert_eq!(
            labels,
            ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>()
        );
        assert_eq!(weights, vec![1.0; 8]);
        assert_eq!(retained_response_tokens, 3);
    }

    #[test]
    fn response_boundary_weight_never_boosts_eos() {
        let options = TextDatasetOptions {
            train_prompt_tokens: false,
            prompt_loss_weight: 0.1,
            response_loss_weight: 1.0,
            response_boundary_loss_weight: 2.0,
            response_boundary_tokens: 32,
            ..TextDatasetOptions::default()
        };
        let (ids, labels, weights, retained_response_tokens) =
            compose_prompt_response(&[10, 11], &[20, 21], 99, &options)
                .expect("row should compose");

        assert_eq!(ids, vec![10, 11, 20, 21, 99]);
        assert_eq!(labels, vec![-100, -100, 20, 21, 99]);
        assert_eq!(weights, vec![0.0, 0.0, 2.0, 2.0, 1.0]);
        assert_eq!(retained_response_tokens, 2);
    }

    #[test]
    fn root_text_schema_detection_covers_content_and_question_answer() {
        let options = TextDatasetOptions::default();
        let content = json!({"content": "plain document"});
        assert_eq!(
            resolve_text_sample_columns(&content, &options),
            (Some("content".to_string()), None, None)
        );

        let qa = json!({"question": "2+2?", "answer": "4"});
        assert_eq!(
            resolve_text_sample_columns(&qa, &options),
            (
                None,
                Some("question".to_string()),
                Some("answer".to_string())
            )
        );

        let alpaca = TextDatasetOptions {
            alpaca: true,
            ..TextDatasetOptions::default()
        };
        assert_eq!(
            resolve_text_sample_columns(&json!({"instruction": "go", "output": "done"}), &alpaca),
            (
                None,
                Some("instruction".to_string()),
                Some("output".to_string())
            )
        );
    }

    #[test]
    fn local_tokenizer_assets_are_installed_into_output_and_checkpoints() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hierarchos-native-cli-tokenizer-assets-{}-{nonce}",
            std::process::id()
        ));
        let source = root.join("tokenizer-source");
        let output = root.join("model-output");
        let checkpoint = output.join("checkpoint-epoch-1-step-1");
        fs::create_dir_all(&source).expect("create tokenizer source");
        fs::create_dir_all(&checkpoint).expect("create checkpoint output");
        fs::write(source.join("tokenizer.json"), b"tokenizer-fixture")
            .expect("write tokenizer fixture");
        fs::write(source.join("special_tokens_map.json"), b"special-fixture")
            .expect("write tokenizer sidecar fixture");

        install_local_tokenizer_assets(&source, &output).expect("install output tokenizer assets");
        install_local_tokenizer_assets_into_checkpoints(&source, &output)
            .expect("install checkpoint tokenizer assets");

        assert_eq!(
            fs::read(output.join("tokenizer.json")).expect("read output tokenizer"),
            b"tokenizer-fixture"
        );
        assert_eq!(
            fs::read(checkpoint.join("special_tokens_map.json"))
                .expect("read checkpoint tokenizer sidecar"),
            b"special-fixture"
        );
        fs::remove_dir_all(&root).expect("remove tokenizer asset fixture");
    }

    #[test]
    fn package_vocab_size_accepts_native_config_contract() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hierarchos-native-cli-vocab-config-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create vocab config fixture");
        fs::write(
            root.join("hierarchos_rust_config.json"),
            br#"{"architecture_contract":{"vocab_size":50257}}"#,
        )
        .expect("write vocab config fixture");
        assert_eq!(package_vocab_size(&root).expect("read vocab size"), 50_257);
        fs::remove_dir_all(&root).expect("remove vocab config fixture");
    }

    #[test]
    fn contract_staging_updates_nested_and_flat_configs_without_mutating_source() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hierarchos-native-cli-contract-staging-{}-{nonce}",
            std::process::id()
        ));
        let source = root.join("source");
        let output = root.join("output");

        let mut bootstrap = NativeBootstrapConfig::for_vocab(64);
        bootstrap.context_dim = 32;
        bootstrap.h_hidden = 32;
        bootstrap.l_hidden = 32;
        bootstrap.persistent_dim = 8;
        bootstrap.ltm_slots = 16;
        bootstrap.ltm_key_dim = 8;
        bootstrap.ltm_val_dim = 8;
        bootstrap.ltm_topk = 2;
        bootstrap.h_stride = 2;
        bootstrap.max_h_steps = 3;
        bootstrap.max_l_steps = 2;
        bootstrap.rwkv_head_size = 32;
        bootstrap.token_adapter_rank = 32;
        bootstrap.rosa_max_context = 8;
        initialize_model_package(&source, &bootstrap).expect("bootstrap source package");

        // Match the external compatibility package shape used by existing
        // Hierarchos exports: Rust config owns the nested canonical contract,
        // while the compatibility config is a flat projection plus the hash.
        let compatibility_path = source.join("hierarchos_config.json");
        let mut compatibility: Value = serde_json::from_slice(
            &fs::read(&compatibility_path).expect("read compatibility config"),
        )
        .expect("decode compatibility config");
        compatibility
            .as_object_mut()
            .expect("compatibility config object")
            .remove("architecture_contract");
        fs::write(
            &compatibility_path,
            serde_json::to_vec_pretty(&compatibility).expect("encode flat compatibility config"),
        )
        .expect("write flat compatibility config");

        let source_before = fs::read(source.join("hierarchos_rust_config.json"))
            .expect("read source config before staging");
        let mut overrides = ArchitectureContractOverrides::new();
        capture_contract_number(
            &mut overrides,
            "drift_delta_scale",
            "0.35",
            "--drift-delta-scale",
        )
        .expect("capture drift override");
        capture_contract_number(
            &mut overrides,
            "ponder_loss_weight",
            "0.003",
            "--ponder-loss-weight",
        )
        .expect("capture ponder override");

        let staged = stage_model_with_contract_overrides(&source, &output, &overrides)
            .expect("stage package with contract overrides");
        assert_ne!(staged, source);
        assert_eq!(
            fs::read(source.join("hierarchos_rust_config.json"))
                .expect("read source config after staging"),
            source_before,
            "contract staging must never mutate source model metadata"
        );
        assert!(staged.join("model.safetensors").is_file());

        let rust_config: Value = serde_json::from_slice(
            &fs::read(staged.join("hierarchos_rust_config.json")).expect("read staged Rust config"),
        )
        .expect("decode staged Rust config");
        let compatibility_config: Value = serde_json::from_slice(
            &fs::read(staged.join("hierarchos_config.json"))
                .expect("read staged compatibility config"),
        )
        .expect("decode staged compatibility config");
        assert_eq!(
            rust_config.pointer("/architecture_contract/drift_delta_scale"),
            Some(&json!(0.35))
        );
        assert_eq!(rust_config.get("drift_delta_scale"), Some(&json!(0.35)));
        assert_eq!(
            compatibility_config.get("drift_delta_scale"),
            Some(&json!(0.35))
        );
        assert!(compatibility_config.get("architecture_contract").is_none());
        assert_eq!(
            rust_config.get("architecture_contract_sha256"),
            compatibility_config.get("architecture_contract_sha256")
        );
        hierarchos_inference::ModelConfig::from_model_dir(&staged)
            .expect("staged package must satisfy native runtime contract");

        let mut matching = ArchitectureContractOverrides::new();
        capture_contract_number(
            &mut matching,
            "drift_delta_scale",
            "0.35",
            "--drift-delta-scale",
        )
        .expect("capture matching resume override");
        validate_exact_resume_contract_overrides(&staged, &matching)
            .expect("exact resume should accept a repeated matching contract value");
        capture_contract_number(
            &mut matching,
            "drift_delta_scale",
            "0.5",
            "--drift-delta-scale",
        )
        .expect("capture mismatching resume override");
        let error = validate_exact_resume_contract_overrides(&staged, &matching)
            .expect_err("exact resume must reject contract drift");
        assert!(error.contains("exact native resume forbids changing"));

        fs::remove_dir_all(&root).expect("remove contract staging fixture");
    }

    #[test]
    fn hugging_face_repo_and_file_validation_is_fail_closed() {
        assert!(validate_hf_repo_id("openai-community/gpt2").is_ok());
        assert!(validate_hf_repo_id("owner").is_err());
        assert!(validate_hf_repo_id("owner/repo/extra").is_err());
        assert!(validate_hf_repo_id("../repo").is_err());
        assert!(validate_hf_repo_id("owner\\repo").is_err());

        assert!(validate_hf_relative_path("data/train.jsonl").is_ok());
        assert!(validate_hf_relative_path("train.jsonl").is_ok());
        assert!(validate_hf_relative_path("../train.jsonl").is_err());
        assert!(validate_hf_relative_path("data/../train.jsonl").is_err());
        assert!(validate_hf_relative_path("/train.jsonl").is_err());
    }

    #[test]
    fn hugging_face_url_components_are_encoded_without_flattening_file_paths() {
        assert_eq!(hf_encode_path("owner/model"), "owner/model");
        assert_eq!(
            hf_encode_path("data/train set.jsonl"),
            "data/train%20set.jsonl"
        );
        assert_eq!(hf_encode_component("feature/native"), "feature%2Fnative");
    }

    #[test]
    fn transformer_dropout_override_is_explicit_and_covers_nested_mpt_field() {
        let mut config = json!({
            "model_type": "gpt2",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.2,
            "attn_pdrop": 0.3,
            "attention_dropout": 0.4,
            "layerdrop": 0.5,
            "unchanged": 7,
            "attn_config": {
                "attn_pdrop": 0.6,
                "other": true
            }
        });
        assert_eq!(disable_transformer_dropout_fields(&mut config), 6);
        for field in [
            "resid_pdrop",
            "embd_pdrop",
            "attn_pdrop",
            "attention_dropout",
            "layerdrop",
        ] {
            assert_eq!(config[field], json!(0.0));
        }
        assert_eq!(config["attn_config"]["attn_pdrop"], json!(0.0));
        assert_eq!(config["unchanged"], json!(7));
        assert_eq!(config["attn_config"]["other"], json!(true));
        assert_eq!(disable_transformer_dropout_fields(&mut config), 0);
    }

    #[test]
    fn hugging_face_dataset_split_matching_covers_single_sharded_jsonl_and_parquet() {
        assert!(is_hf_jsonl_path("data/train.jsonl"));
        assert!(is_hf_jsonl_path("data/train-00000-of-00002.ndjson"));
        assert!(!is_hf_jsonl_path("data/train.parquet"));
        assert!(is_hf_parquet_path("data/train.parquet"));
        assert!(is_hf_native_dataset_path("data/train.parquet"));
        assert!(is_hf_native_dataset_path("data/train.jsonl"));
        assert!(!is_hf_native_dataset_path("data/train.csv"));
        assert!(hf_path_matches_split("data/train.jsonl", "train"));
        assert!(hf_path_matches_split(
            "config/train-00000-of-00002.jsonl",
            "train"
        ));
        assert!(hf_path_matches_split(
            "config/train-00000-of-00002.parquet",
            "train"
        ));
        assert!(hf_path_matches_split("train/part-000.jsonl", "train"));
        assert!(!hf_path_matches_split("data/validation.jsonl", "train"));
        assert!(hf_path_looks_like_ordered_shard(
            "data/train-00000-of-00002.jsonl"
        ));
        assert!(hf_path_looks_like_ordered_shard(
            "data/train-00000-of-00002.parquet"
        ));
        assert!(!hf_path_looks_like_ordered_shard("data/train.jsonl"));
        assert!(hf_path_has_component(
            "config-a/train-00000.jsonl",
            "config-a"
        ));
    }

    #[test]
    fn parquet_dataset_is_materialized_as_jsonl_without_python() {
        use std::sync::Arc;

        use parquet::{
            data_type::{ByteArray, ByteArrayType, Int64Type},
            file::writer::SerializedFileWriter,
            schema::parser::parse_message_type,
        };

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hierarchos-native-parquet-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create parquet test directory");
        let parquet_path = root.join("train.parquet");
        let schema = Arc::new(
            parse_message_type(
                "message schema { REQUIRED BYTE_ARRAY text (UTF8); REQUIRED INT64 id; }",
            )
            .expect("parse parquet test schema"),
        );
        let file = File::create(&parquet_path).expect("create parquet fixture");
        let mut writer = SerializedFileWriter::new(file, schema, Default::default())
            .expect("create parquet writer");
        let mut row_group = writer.next_row_group().expect("create row group");
        {
            let mut column = row_group
                .next_column()
                .expect("read text column")
                .expect("text column exists");
            column
                .typed::<ByteArrayType>()
                .write_batch(
                    &[ByteArray::from("alpha"), ByteArray::from("beta")],
                    None,
                    None,
                )
                .expect("write text column");
            column.close().expect("close text column");
        }
        {
            let mut column = row_group
                .next_column()
                .expect("read id column")
                .expect("id column exists");
            column
                .typed::<Int64Type>()
                .write_batch(&[7, 9], None, None)
                .expect("write id column");
            column.close().expect("close id column");
        }
        assert!(row_group
            .next_column()
            .expect("finish parquet columns")
            .is_none());
        row_group.close().expect("close parquet row group");
        writer.close().expect("close parquet writer");

        let jsonl_path =
            materialize_parquet_jsonl(&parquet_path).expect("materialize parquet fixture as JSONL");
        let lines = fs::read_to_string(&jsonl_path).expect("read converted JSONL");
        let rows = lines
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("parse converted JSON row"))
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["text"], "alpha");
        assert_eq!(rows[0]["id"], 7);
        assert_eq!(rows[1]["text"], "beta");
        assert_eq!(rows[1]["id"], 9);

        fs::remove_dir_all(&root).expect("remove parquet test directory");
    }

    #[test]
    fn transformer_safetensors_index_collects_unique_safe_shards() {
        let index = json!({
            "metadata": {"total_size": 1234},
            "weight_map": {
                "transformer.wte.weight": "model-00001-of-00002.safetensors",
                "transformer.h.0.ln_1.weight": "model-00001-of-00002.safetensors",
                "transformer.h.1.ln_1.weight": "nested/model-00002-of-00002.safetensors"
            }
        });
        assert_eq!(
            transformer_safetensors_shards_from_index_value(&index)
                .expect("valid sharded SafeTensors index"),
            vec![
                "model-00001-of-00002.safetensors".to_owned(),
                "nested/model-00002-of-00002.safetensors".to_owned()
            ]
        );

        let traversal = json!({
            "weight_map": {"transformer.wte.weight": "../escape.safetensors"}
        });
        assert!(transformer_safetensors_shards_from_index_value(&traversal).is_err());

        let wrong_extension = json!({
            "weight_map": {"transformer.wte.weight": "pytorch_model.bin"}
        });
        assert!(transformer_safetensors_shards_from_index_value(&wrong_extension).is_err());

        let windows_separator = json!({
            "weight_map": {"transformer.wte.weight": "weights\\model.safetensors"}
        });
        assert!(transformer_safetensors_shards_from_index_value(&windows_separator).is_err());
    }

    #[test]
    fn transformer_chunking_builds_shifted_causal_targets_and_padding() {
        let value = json!({
            "input_ids": [3, 7, 2, 11, 9, 4],
            "labels": [3, 7, -100, 11, 9, 4],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "loss_weights": [0.5, 1.0, 1.0, 2.0, 1.0, 3.0]
        });
        let chunks = transformer_chunks_from_value(&value, 4, 8).expect("chunk tokenized row");
        assert_eq!(chunks.len(), 2);

        assert_eq!(chunks[0].input_ids, vec![3, 7, 2, 11]);
        assert_eq!(chunks[0].targets, vec![7, u32::MAX, 11, 9]);
        assert_eq!(chunks[0].attention_mask, vec![1.0; 4]);
        assert_eq!(chunks[0].loss_weights, vec![1.0, 0.0, 2.0, 1.0]);

        assert_eq!(chunks[1].input_ids, vec![9, 0, 0, 0]);
        assert_eq!(chunks[1].targets, vec![4, u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(chunks[1].attention_mask, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(chunks[1].loss_weights, vec![3.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn transformer_chunking_attaches_cross_attention_encoder_context() {
        let value = json!({
            "input_ids": [3, 7, 9],
            "labels": [3, 7, 9],
            "encoder_hidden_states": [[0.1, 0.2], [0.3, 0.4]],
            "encoder_attention_mask": [1, 0]
        });
        let chunks = transformer_chunks_from_value(&value, 4, 2)
            .expect("chunk cross-attention tokenized row");
        let context = chunks[0]
            .encoder_context
            .as_ref()
            .expect("cross-attention context should be attached");
        assert_eq!(context.seq_len, 2);
        assert_eq!(context.hidden_states, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(context.attention_mask, vec![1.0, 0.0]);
    }

    #[test]
    fn transformer_encoder_context_rejects_hidden_width_mismatch() {
        let value = json!({
            "input_ids": [3, 7, 9],
            "encoder_hidden_states": [0.1, 0.2, 0.3]
        });
        let error = transformer_chunks_from_value(&value, 4, 2)
            .expect_err("encoder hidden width mismatch should fail");
        assert!(error.contains("positive multiple"));
    }

    #[test]
    fn failed_ckpt_to_inference_does_not_publish_partial_output() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hierarchos-native-cli-ckpt-failure-{}-{nonce}",
            std::process::id()
        ));
        let source = root.join("source");
        let output = root.join("output");
        fs::create_dir_all(&source).expect("create test source package");
        fs::write(source.join("hierarchos_rust_config.json"), b"{}\n")
            .expect("write test Rust config");
        fs::write(source.join("hierarchos_config.json"), b"{}\n")
            .expect("write test compatibility config");
        fs::write(source.join("model.safetensors"), b"invalid-safetensors")
            .expect("write invalid model fixture");

        let args = VecDeque::from(vec![
            OsString::from("--ckpt-input"),
            source.as_os_str().to_os_string(),
            OsString::from("--inf-output"),
            output.as_os_str().to_os_string(),
        ]);
        let error = run_ckpt_to_inference(args).expect_err("invalid package must fail validation");
        assert!(!error.is_empty());
        assert!(
            !output.exists(),
            "failed ckpt-2-inf must not publish a partial output directory"
        );
        let staging_prefix = ".output.native-inference-staging-";
        let leaked_staging = fs::read_dir(&root)
            .expect("read test root")
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(staging_prefix)
            });
        assert!(
            !leaked_staging,
            "failed ckpt-2-inf must remove staging data"
        );

        fs::remove_dir_all(&root).expect("remove ckpt-2-inf regression fixture");
    }
}
