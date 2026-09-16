use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use regex::Regex;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::checkpoint::replace_f32_tensor_values;

pub const HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME: &str = "adapter_config.json";
pub const HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME: &str = "adapter_model.safetensors";
pub const HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME: &str = "hierarchos_adapter_manifest.json";

const HIERARCHOS_LORA_ADAPTER_MANIFEST_FORMAT: &str = "hierarchos-peft-lora-v1";
const HIERARCHOS_LORA_ADAPTER_MANIFEST_VERSION: u64 = 1;
const PEFT_STATE_PREFIX: &str = "base_model.model.";
const LORA_A_SUFFIX: &str = ".lora_A.weight";
const LORA_B_SUFFIX: &str = ".lora_B.weight";
const LORA_B_BIAS_SUFFIX: &str = ".lora_B.bias";
const LORA_MAGNITUDE_SUFFIX: &str = ".lora_magnitude_vector";
const LORA_MAGNITUDE_WEIGHT_SUFFIX: &str = ".lora_magnitude_vector.weight";
const PEFT_PARAMETER_BASE_LAYER_SUFFIX: &str = ".base_layer";

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModuleSelector {
    Regex(String),
    Modules(Vec<String>),
}

impl Default for ModuleSelector {
    fn default() -> Self {
        Self::Modules(Vec::new())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosNativeLoraMergeReport {
    pub merged_lora_modules: usize,
    pub replaced_module_tensors: usize,
    pub base_checkpoint_sha256: String,
    pub adapter_checkpoint_sha256: String,
    pub architecture_contract_sha256: String,
}

#[derive(Debug, Deserialize)]
struct LoraAdapterConfig {
    #[serde(default)]
    peft_type: String,
    #[serde(default)]
    task_type: String,
    r: usize,
    lora_alpha: f64,
    #[serde(default)]
    lora_dropout: f64,
    #[serde(default, deserialize_with = "deserialize_module_selector")]
    target_modules: ModuleSelector,
    #[serde(default, deserialize_with = "deserialize_module_selector")]
    exclude_modules: ModuleSelector,
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    #[serde(default)]
    bias: String,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    use_qalora: bool,
    #[serde(default)]
    lora_bias: bool,
    #[serde(default)]
    target_parameters: Option<Vec<String>>,
    #[serde(default)]
    trainable_token_indices: Option<serde_json::Value>,
    #[serde(default)]
    alora_invocation_tokens: Option<serde_json::Value>,
    #[serde(default)]
    arrow_config: Option<serde_json::Value>,
    #[serde(default)]
    layer_replication: Option<serde_json::Value>,
    #[serde(default, rename = "corda_config")]
    _corda_config: Option<serde_json::Value>,
    #[serde(default)]
    rank_pattern: BTreeMap<String, usize>,
    #[serde(default)]
    alpha_pattern: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct ManifestFileBinding {
    filename: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct LoraAdapterManifest {
    manifest_version: u64,
    format: String,
    base_checkpoint: ManifestFileBinding,
    architecture_contract_sha256: String,
    adapter_files: BTreeMap<String, String>,
}

#[derive(Debug)]
struct AdapterPair {
    module_name: String,
    nested_parameter_wrapper: bool,
    a_key: String,
    b_key: String,
    b_bias_key: Option<String>,
    magnitude_key: Option<String>,
}

#[derive(Debug)]
struct AdapterReplacement {
    adapter_key: String,
    target_name: String,
}

#[derive(Debug)]
struct TrainableTokenReplacement {
    adapter_key: String,
    module_name: String,
}

/// Merge the exact SafeTensors PEFT-LoRA package emitted by Hierarchos into a
/// standalone canonical model checkpoint without importing Python, PEFT, or
/// PyTorch. The model tensor layout remains the shared row-major ABI consumed
/// by native Rust inference, the Vulkan trainer, and compatible CUDA readers.
///
/// This is deliberately a package/checkpoint operation rather than a training
/// primitive. Training remains Vulkan-only; the small A/B merge is deterministic
/// native Rust arithmetic and is performed once when publishing an adapter.
pub fn merge_hierarchos_lora_safetensors(
    base_weights: &Path,
    adapter_dir: &Path,
    destination: &Path,
) -> Result<HierarchosNativeLoraMergeReport> {
    if !base_weights.is_file() {
        bail!(
            "base model weights do not exist: {}",
            base_weights.display()
        );
    }
    if !adapter_dir.is_dir() {
        bail!(
            "LoRA adapter directory does not exist: {}",
            adapter_dir.display()
        );
    }
    if base_weights == destination {
        bail!("native LoRA merge requires a distinct destination checkpoint");
    }

    reject_pickle_adapter_files(adapter_dir)?;
    let config_path = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME);
    let adapter_weights = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME);
    let manifest_path = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME);
    let config = read_adapter_config(&config_path)?;
    validate_adapter_config(&config)?;
    let base_sha256 = sha256_file(base_weights)?;
    let adapter_sha256 = sha256_file(&adapter_weights)?;
    let architecture_contract_sha256 = validate_bound_manifest(
        base_weights,
        &config_path,
        &adapter_weights,
        &manifest_path,
        &base_sha256,
    )?;

    let base_bytes = fs::read(base_weights)
        .with_context(|| format!("reading base model {}", base_weights.display()))?;
    let base_tensors = SafeTensors::deserialize(&base_bytes)
        .with_context(|| format!("parsing base model {}", base_weights.display()))?;
    let adapter_bytes = fs::read(&adapter_weights)
        .with_context(|| format!("reading adapter {}", adapter_weights.display()))?;
    let adapter_tensors = SafeTensors::deserialize(&adapter_bytes)
        .with_context(|| format!("parsing adapter {}", adapter_weights.display()))?;

    let (pairs, adapter_replacements, trainable_token_replacements) =
        discover_adapter_tensors(&adapter_tensors, &config)?;
    let mut replacements = BTreeMap::<String, Vec<f32>>::new();
    let mut lora_bias_deltas = BTreeMap::<String, Vec<f32>>::new();

    for pair in &pairs {
        let a = adapter_tensors.tensor(&pair.a_key)?;
        let b = adapter_tensors.tensor(&pair.b_key)?;
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            bail!(
                "LoRA module {:?} requires rank-2 A/B tensors; got A={a_shape:?} B={b_shape:?}",
                pair.module_name
            );
        }
        let parameter_target = resolve_target_parameter(pair, &config, &base_tensors)
            .with_context(|| {
                format!(
                    "resolving PEFT target for adapter pair {:?}",
                    pair.module_name
                )
            })?;
        let target_name = parameter_target
            .clone()
            .unwrap_or_else(|| format!("{}.weight", pair.module_name));
        let pattern_name = parameter_target.as_deref().unwrap_or(&pair.module_name);
        let expected_rank = pattern_value(pattern_name, &config.rank_pattern)
            .copied()
            .unwrap_or(config.r);
        let alpha = pattern_value(pattern_name, &config.alpha_pattern)
            .copied()
            .unwrap_or(config.lora_alpha);
        if !alpha.is_finite() || alpha <= 0.0 {
            bail!("LoRA target {pattern_name:?} has invalid alpha {alpha}");
        }
        let scale = lora_scale(alpha, expected_rank, config.use_rslora, pattern_name)?;
        let base = base_tensors.tensor(&target_name).with_context(|| {
            format!(
                "adapter targets tensor {target_name:?}, which is absent from {}",
                base_weights.display()
            )
        })?;
        let base_shape = base.shape().to_vec();
        let a_values = decode_float_tensor(&pair.a_key, &a)?;
        let b_values = decode_float_tensor(&pair.b_key, &b)?;
        let mut merged = decode_float_tensor(&target_name, &base)?;

        if parameter_target.is_some() {
            if base_shape.len() != 3 {
                bail!(
                    "PEFT target_parameters tensor {target_name:?} must be rank-3 for packed expert merge; got {base_shape:?}"
                );
            }
            let num_experts = base_shape[0];
            let in_dim = base_shape[1];
            let out_dim = base_shape[2];
            let expanded_rank = expected_rank
                .checked_mul(num_experts)
                .context("packed target_parameters rank overflow")?;
            if a_shape != [expanded_rank, in_dim] || b_shape != [out_dim, expanded_rank] {
                bail!(
                    "PEFT target_parameters tensor {target_name:?} has incompatible adapter geometry A={a_shape:?} B={b_shape:?}; expected A=[{expanded_rank}, {in_dim}] B=[{out_dim}, {expanded_rank}]"
                );
            }
            add_packed_parameter_lora_delta(
                &mut merged,
                &a_values,
                &b_values,
                num_experts,
                in_dim,
                out_dim,
                expected_rank,
                scale,
            )?;
        } else {
            let rank = a_shape[0];
            if rank == 0 || b_shape[1] != rank {
                bail!(
                    "LoRA module {:?} has inconsistent rank geometry A={a_shape:?} B={b_shape:?}",
                    pair.module_name
                );
            }
            if rank != expected_rank {
                bail!(
                    "LoRA module {:?} has rank {rank}, but adapter config requires {expected_rank}",
                    pair.module_name
                );
            }
            let out_dim = b_shape[0];
            let in_dim = a_shape[1];
            let expected_shape = if config.fan_in_fan_out {
                vec![in_dim, out_dim]
            } else {
                vec![out_dim, in_dim]
            };
            if base_shape != expected_shape {
                bail!(
                    "LoRA module {:?} resolves to base shape {base_shape:?}; expected {expected_shape:?} from A={a_shape:?} B={b_shape:?}",
                    pair.module_name
                );
            }
            add_lora_delta(
                &mut merged,
                &a_values,
                &b_values,
                out_dim,
                rank,
                in_dim,
                scale,
                config.fan_in_fan_out,
            )?;
            if let Some(magnitude_key) = pair.magnitude_key.as_deref() {
                let magnitude_tensor = adapter_tensors.tensor(magnitude_key)?;
                if magnitude_tensor.shape() != [out_dim] {
                    bail!(
                        "DoRA magnitude tensor {magnitude_key:?} has shape {:?}; expected [{out_dim}]",
                        magnitude_tensor.shape()
                    );
                }
                let magnitude = decode_float_tensor(magnitude_key, &magnitude_tensor)?;
                apply_dora_magnitude(
                    &mut merged,
                    &magnitude,
                    in_dim,
                    out_dim,
                    config.fan_in_fan_out,
                )?;
            }
            if let Some(b_bias_key) = pair.b_bias_key.as_deref() {
                let b_bias_tensor = adapter_tensors.tensor(b_bias_key)?;
                if b_bias_tensor.shape() != [out_dim] {
                    bail!(
                        "LoRA B-bias tensor {b_bias_key:?} has shape {:?}; expected [{out_dim}]",
                        b_bias_tensor.shape()
                    );
                }
                let bias_name = format!("{}.bias", pair.module_name);
                if base_tensors.tensor(&bias_name).is_err() {
                    bail!(
                        "impossible to merge lora_bias=true for {:?}: base layer has no bias tensor {bias_name:?}",
                        pair.module_name
                    );
                }
                let mut delta = decode_float_tensor(b_bias_key, &b_bias_tensor)?;
                for value in &mut delta {
                    *value *= scale;
                    if !value.is_finite() {
                        bail!("LoRA B-bias merge produced a non-finite value");
                    }
                }
                if lora_bias_deltas.insert(bias_name.clone(), delta).is_some() {
                    bail!("adapter attempts to merge LoRA B bias into {bias_name:?} twice");
                }
            }
        }
        if replacements.insert(target_name.clone(), merged).is_some() {
            bail!("adapter attempts to replace base tensor {target_name:?} twice");
        }
    }

    for replacement in &adapter_replacements {
        let saved = adapter_tensors.tensor(&replacement.adapter_key)?;
        let base = base_tensors
            .tensor(&replacement.target_name)
            .with_context(|| {
                format!(
                    "adapter replacement tensor {:?} resolves to missing base tensor {:?}",
                    replacement.adapter_key, replacement.target_name
                )
            })?;
        if saved.shape() != base.shape() {
            bail!(
                "adapter replacement tensor {:?} has shape {:?}; base tensor {:?} has shape {:?}",
                replacement.adapter_key,
                saved.shape(),
                replacement.target_name,
                base.shape()
            );
        }
        let values = decode_float_tensor(&replacement.adapter_key, &saved)?;
        if replacements
            .insert(replacement.target_name.clone(), values)
            .is_some()
        {
            bail!(
                "adapter attempts to replace base tensor {:?} twice",
                replacement.target_name
            );
        }
    }

    for replacement in &trainable_token_replacements {
        let saved = adapter_tensors.tensor(&replacement.adapter_key)?;
        let token_indices = trainable_token_indices_for_module(&config, &replacement.module_name)?;
        let target_name = format!("{}.weight", replacement.module_name);
        let base = base_tensors.tensor(&target_name).with_context(|| {
            format!(
                "trainable-token adapter tensor {:?} resolves to missing base embedding tensor {:?}",
                replacement.adapter_key, target_name
            )
        })?;
        let base_shape = base.shape();
        if base_shape.len() != 2 {
            bail!(
                "trainable-token base tensor {target_name:?} must be rank-2 [vocab, hidden], got {base_shape:?}"
            );
        }
        let expected_shape = [token_indices.len(), base_shape[1]];
        if saved.shape() != expected_shape {
            bail!(
                "trainable-token adapter tensor {:?} has shape {:?}; expected {:?} from trainable_token_indices",
                replacement.adapter_key,
                saved.shape(),
                expected_shape
            );
        }
        if let Some(&row) = token_indices.iter().find(|&&row| row >= base_shape[0]) {
            bail!(
                "trainable_token_indices row {row} is outside base embedding vocabulary {} for {target_name:?}",
                base_shape[0]
            );
        }
        let saved_values = decode_float_tensor(&replacement.adapter_key, &saved)?;
        let hidden = base_shape[1];
        let values = if let Some(values) = replacements.get_mut(&target_name) {
            values
        } else {
            replacements.insert(
                target_name.clone(),
                decode_float_tensor(&target_name, &base)?,
            );
            replacements
                .get_mut(&target_name)
                .expect("trainable-token base tensor was inserted above")
        };
        for (source_row, &target_row) in token_indices.iter().enumerate() {
            let source_start = source_row * hidden;
            let target_start = target_row * hidden;
            values[target_start..target_start + hidden]
                .copy_from_slice(&saved_values[source_start..source_start + hidden]);
        }
    }

    for (bias_name, delta) in lora_bias_deltas {
        if let Some(values) = replacements.get_mut(&bias_name) {
            if values.len() != delta.len() {
                bail!(
                    "saved bias tensor {bias_name:?} length {} does not match LoRA B-bias length {}",
                    values.len(),
                    delta.len()
                );
            }
            for (value, add) in values.iter_mut().zip(delta) {
                *value += add;
                if !value.is_finite() {
                    bail!("LoRA B-bias merge produced a non-finite bias");
                }
            }
        } else {
            let base_bias = base_tensors.tensor(&bias_name).with_context(|| {
                format!("LoRA B bias resolves to missing base tensor {bias_name:?}")
            })?;
            let mut values = decode_float_tensor(&bias_name, &base_bias)?;
            if values.len() != delta.len() {
                bail!(
                    "base bias tensor {bias_name:?} length {} does not match LoRA B-bias length {}",
                    values.len(),
                    delta.len()
                );
            }
            for (value, add) in values.iter_mut().zip(delta) {
                *value += add;
                if !value.is_finite() {
                    bail!("LoRA B-bias merge produced a non-finite bias");
                }
            }
            replacements.insert(bias_name, values);
        }
    }

    if replacements.is_empty() {
        bail!("LoRA adapter contains no mergeable tensors");
    }
    drop(adapter_tensors);
    drop(adapter_bytes);
    drop(base_tensors);
    drop(base_bytes);

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating merge output directory {}", parent.display()))?;
    }
    let replacement_refs = replacements
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect::<Vec<_>>();
    replace_f32_tensor_values(base_weights, destination, &replacement_refs)
        .with_context(|| format!("writing merged model {}", destination.display()))?;

    Ok(HierarchosNativeLoraMergeReport {
        merged_lora_modules: pairs.len(),
        replaced_module_tensors: adapter_replacements.len() + trainable_token_replacements.len(),
        base_checkpoint_sha256: base_sha256,
        adapter_checkpoint_sha256: adapter_sha256,
        architecture_contract_sha256,
    })
}

fn read_adapter_config(path: &Path) -> Result<LoraAdapterConfig> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn deserialize_module_selector<'de, D>(
    deserializer: D,
) -> std::result::Result<ModuleSelector, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<ModuleSelector>::deserialize(deserializer)?.unwrap_or_default())
}

fn validate_adapter_config(config: &LoraAdapterConfig) -> Result<()> {
    if !config.peft_type.eq_ignore_ascii_case("LORA") {
        bail!(
            "unsupported PEFT adapter type {:?}; expected LORA",
            config.peft_type
        );
    }
    if !config.task_type.is_empty()
        && ![
            "SEQ_CLS",
            "SEQ_2_SEQ_LM",
            "CAUSAL_LM",
            "TOKEN_CLS",
            "QUESTION_ANS",
            "FEATURE_EXTRACTION",
        ]
        .iter()
        .any(|task_type| config.task_type.eq_ignore_ascii_case(task_type))
    {
        bail!(
            "unsupported PEFT task type {:?}; expected a current PEFT task type",
            config.task_type
        );
    }
    if config.r == 0 || !config.lora_alpha.is_finite() || config.lora_alpha <= 0.0 {
        bail!("LoRA rank and alpha must both be finite and positive");
    }
    if !config.lora_dropout.is_finite() || config.lora_dropout < 0.0 || config.lora_dropout >= 1.0 {
        bail!("LoRA dropout must be finite and in [0, 1)");
    }
    if !config.bias.is_empty()
        && !config.bias.eq_ignore_ascii_case("none")
        && !config.bias.eq_ignore_ascii_case("lora_only")
        && !config.bias.eq_ignore_ascii_case("all")
    {
        bail!("PEFT LoRA bias must be one of 'none', 'lora_only', or 'all'");
    }
    let target_parameters = config.target_parameters.as_deref().unwrap_or_default();
    if module_selector_is_empty(&config.target_modules) && target_parameters.is_empty() {
        bail!("PEFT LoRA adapter has neither target_modules nor target_parameters");
    }
    validate_module_selector("target_modules", &config.target_modules)?;
    validate_module_selector("exclude_modules", &config.exclude_modules)?;
    if target_parameters
        .iter()
        .any(|parameter| parameter.trim().is_empty())
    {
        bail!("PEFT target_parameters entries must not be empty");
    }
    if target_parameters.len()
        != target_parameters
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .len()
    {
        bail!("PEFT target_parameters contains duplicate parameter names");
    }
    if config.lora_bias && config.use_dora {
        bail!("PEFT lora_bias=true cannot be combined with use_dora=true");
    }
    if !target_parameters.is_empty() && config.lora_bias {
        bail!("PEFT target_parameters does not support lora_bias=true");
    }
    if !target_parameters.is_empty() && config.use_dora {
        bail!("PEFT target_parameters does not support use_dora=true");
    }
    let modules_to_save = config.modules_to_save.as_deref().unwrap_or_default();
    if modules_to_save
        .iter()
        .any(|module| module.trim().is_empty())
    {
        bail!("PEFT modules_to_save entries must not be empty");
    }
    if modules_to_save.len()
        != modules_to_save
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .len()
    {
        bail!("PEFT modules_to_save contains duplicate module names");
    }
    let unsupported = [
        ("use_qalora", config.use_qalora),
        (
            "alora_invocation_tokens",
            nonempty_json_option(&config.alora_invocation_tokens),
        ),
        ("arrow_config", nonempty_json_option(&config.arrow_config)),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>();
    if !unsupported.is_empty() {
        bail!(
            "PEFT adapter uses non-mergeable LoRA feature(s): {}",
            unsupported.join(", ")
        );
    }
    if nonempty_json_option(&config.layer_replication) {
        bail!(
            "PEFT layer_replication changes the base-model topology and cannot be represented by standalone weight merging; load this adapter through the native runtime graph instead"
        );
    }
    validate_trainable_token_indices(config)?;
    for (name, rank) in &config.rank_pattern {
        if name.trim().is_empty() || *rank == 0 {
            bail!("LoRA rank_pattern contains an empty name or zero rank");
        }
    }
    for (name, alpha) in &config.alpha_pattern {
        if name.trim().is_empty() || !alpha.is_finite() || *alpha <= 0.0 {
            bail!("LoRA alpha_pattern contains an empty name or invalid alpha");
        }
    }
    Ok(())
}

fn nonempty_json_option(value: &Option<serde_json::Value>) -> bool {
    match value {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(false)) => false,
        Some(serde_json::Value::Array(values)) => !values.is_empty(),
        Some(serde_json::Value::Object(values)) => !values.is_empty(),
        Some(serde_json::Value::String(value)) => !value.is_empty(),
        Some(_) => true,
    }
}

fn parse_trainable_token_indices(value: &serde_json::Value, label: &str) -> Result<Vec<usize>> {
    let values = value.as_array().with_context(|| {
        format!("PEFT trainable_token_indices {label} must be an integer array")
    })?;
    if values.is_empty() {
        bail!("PEFT trainable_token_indices {label} must not be empty");
    }
    let mut indices = Vec::with_capacity(values.len());
    let mut unique = BTreeSet::new();
    for value in values {
        let raw = value.as_u64().with_context(|| {
            format!("PEFT trainable_token_indices {label} contains a non-negative integer")
        })?;
        let index = usize::try_from(raw)
            .with_context(|| format!("PEFT trainable_token_indices {label} exceeds usize"))?;
        if !unique.insert(index) {
            bail!("PEFT trainable_token_indices {label} contains duplicate token id {index}");
        }
        indices.push(index);
    }
    Ok(indices)
}

fn validate_trainable_token_indices(config: &LoraAdapterConfig) -> Result<()> {
    let Some(value) = config.trainable_token_indices.as_ref() else {
        return Ok(());
    };
    match value {
        serde_json::Value::Null => Ok(()),
        serde_json::Value::Array(_) => {
            let _ = parse_trainable_token_indices(value, "list")?;
            Ok(())
        }
        serde_json::Value::Object(modules) => {
            if modules.is_empty() {
                bail!("PEFT trainable_token_indices module map must not be empty");
            }
            for (module, indices) in modules {
                if module.trim().is_empty() {
                    bail!("PEFT trainable_token_indices module names must not be empty");
                }
                let _ = parse_trainable_token_indices(indices, &format!("for module {module:?}"))?;
            }
            Ok(())
        }
        _ => {
            bail!("PEFT trainable_token_indices must be an integer array or module-to-indices map")
        }
    }
}

fn trainable_token_indices_for_module(
    config: &LoraAdapterConfig,
    module_name: &str,
) -> Result<Vec<usize>> {
    let value = config.trainable_token_indices.as_ref().with_context(|| {
        format!(
            "adapter contains trainable-token tensor for {module_name:?} but adapter_config.json has no trainable_token_indices"
        )
    })?;
    match value {
        serde_json::Value::Array(_) => parse_trainable_token_indices(value, "list"),
        serde_json::Value::Object(modules) => {
            let matches = modules
                .iter()
                .filter(|(requested, _)| {
                    module_name == requested.as_str()
                        || module_name.ends_with(&format!(".{requested}"))
                })
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                bail!(
                    "trainable-token adapter module {module_name:?} matches {} trainable_token_indices entries; expected exactly one",
                    matches.len()
                );
            }
            let (requested, indices) = matches[0];
            parse_trainable_token_indices(indices, &format!("for module {requested:?}"))
        }
        serde_json::Value::Null => bail!(
            "adapter contains trainable-token tensor for {module_name:?} but trainable_token_indices is null"
        ),
        _ => bail!("PEFT trainable_token_indices must be an integer array or module-to-indices map"),
    }
}

fn reject_pickle_adapter_files(adapter_dir: &Path) -> Result<()> {
    for filename in ["adapter_model.bin", "adapter_model.pt", "pytorch_model.bin"] {
        let path = adapter_dir.join(filename);
        if path.exists() {
            bail!(
                "unsafe pickle-based adapter weights are not supported: {}; re-export as {}",
                path.display(),
                HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME
            );
        }
    }
    Ok(())
}

fn validate_bound_manifest(
    base_weights: &Path,
    config_path: &Path,
    adapter_weights: &Path,
    manifest_path: &Path,
    base_sha256: &str,
) -> Result<String> {
    let checksum_path = PathBuf::from(format!("{}.sha256", manifest_path.display()));
    let expected_manifest_sha = fs::read_to_string(&checksum_path)
        .with_context(|| {
            format!(
                "reading adapter manifest checksum {}",
                checksum_path.display()
            )
        })?
        .split_whitespace()
        .next()
        .context("adapter manifest checksum file is empty")?
        .to_ascii_lowercase();
    if !is_sha256(&expected_manifest_sha) {
        bail!("adapter manifest checksum is not a valid SHA-256 digest");
    }
    let actual_manifest_sha = sha256_file(manifest_path)?;
    if actual_manifest_sha != expected_manifest_sha {
        bail!("adapter manifest SHA-256 verification failed; refusing to merge");
    }
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("reading adapter manifest {}", manifest_path.display()))?;
    let manifest: LoraAdapterManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parsing adapter manifest {}", manifest_path.display()))?;
    if manifest.manifest_version != HIERARCHOS_LORA_ADAPTER_MANIFEST_VERSION {
        bail!(
            "unsupported adapter manifest version {}",
            manifest.manifest_version
        );
    }
    if manifest.format != HIERARCHOS_LORA_ADAPTER_MANIFEST_FORMAT {
        bail!("unsupported adapter manifest format {:?}", manifest.format);
    }
    let base_filename = base_weights
        .file_name()
        .and_then(|name| name.to_str())
        .context("base model checkpoint filename is not valid UTF-8")?;
    if manifest.base_checkpoint.filename != base_filename
        || !manifest
            .base_checkpoint
            .sha256
            .eq_ignore_ascii_case(base_sha256)
    {
        bail!("LoRA adapter is not bound to this exact base checkpoint");
    }
    for (filename, path) in [
        (HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME, config_path),
        (HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME, adapter_weights),
    ] {
        let expected = manifest
            .adapter_files
            .get(filename)
            .with_context(|| format!("adapter manifest has no hash for {filename}"))?;
        let actual = sha256_file(path)?;
        if !expected.eq_ignore_ascii_case(&actual) {
            bail!("LoRA adapter file hash mismatch for {filename}");
        }
    }
    if !is_sha256(&manifest.architecture_contract_sha256) {
        bail!("adapter manifest architecture contract SHA-256 is invalid");
    }
    let base_contract_hash = read_base_architecture_contract_hash(base_weights)?;
    if !manifest
        .architecture_contract_sha256
        .eq_ignore_ascii_case(&base_contract_hash)
    {
        bail!("LoRA adapter architecture contract differs from the selected base");
    }
    Ok(base_contract_hash)
}

fn read_base_architecture_contract_hash(base_weights: &Path) -> Result<String> {
    let model_dir = base_weights.parent().unwrap_or_else(|| Path::new("."));
    for filename in ["hierarchos_rust_config.json", "hierarchos_config.json"] {
        let path = model_dir.join(filename);
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if let Some(hash) = value
            .get("architecture_contract_sha256")
            .and_then(serde_json::Value::as_str)
        {
            let hash = hash.to_ascii_lowercase();
            if is_sha256(&hash) {
                return Ok(hash);
            }
            bail!(
                "{} contains an invalid architecture contract hash",
                path.display()
            );
        }
    }
    bail!(
        "base model package has no architecture_contract_sha256 in hierarchos_rust_config.json or hierarchos_config.json"
    )
}

fn discover_adapter_tensors(
    tensors: &SafeTensors<'_>,
    config: &LoraAdapterConfig,
) -> Result<(
    Vec<AdapterPair>,
    Vec<AdapterReplacement>,
    Vec<TrainableTokenReplacement>,
)> {
    let mut a_keys = BTreeMap::<String, String>::new();
    let mut b_keys = BTreeMap::<String, String>::new();
    let mut b_bias_keys = BTreeMap::<String, String>::new();
    let mut magnitude_keys = BTreeMap::<String, String>::new();
    let mut replacements = Vec::new();
    let mut trainable_token_replacements = Vec::new();
    let modules_to_save = config.modules_to_save.as_deref().unwrap_or_default();

    for key in tensors.names() {
        if !key.starts_with(PEFT_STATE_PREFIX) {
            bail!("unsupported adapter tensor key {key:?}; expected prefix {PEFT_STATE_PREFIX:?}");
        }
        let tensor = tensors.tensor(key)?;
        if tensor.data().is_empty() {
            bail!("PEFT adapter tensor {key:?} is empty");
        }
        if let Some(prefix) = key.strip_suffix(LORA_A_SUFFIX) {
            let raw_module_name = prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .context("LoRA A tensor lost validated PEFT prefix")?;
            let module_name = raw_module_name
                .strip_suffix(PEFT_PARAMETER_BASE_LAYER_SUFFIX)
                .unwrap_or(raw_module_name);
            validate_adapter_pair_target(module_name, config)?;
            if a_keys.insert(prefix.to_string(), key.to_string()).is_some() {
                bail!("duplicate LoRA A tensor for module {module_name:?}");
            }
        } else if let Some(prefix) = key.strip_suffix(LORA_B_SUFFIX) {
            let raw_module_name = prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .context("LoRA B tensor lost validated PEFT prefix")?;
            let module_name = raw_module_name
                .strip_suffix(PEFT_PARAMETER_BASE_LAYER_SUFFIX)
                .unwrap_or(raw_module_name);
            validate_adapter_pair_target(module_name, config)?;
            if b_keys.insert(prefix.to_string(), key.to_string()).is_some() {
                bail!("duplicate LoRA B tensor for module {module_name:?}");
            }
        } else if let Some(prefix) = key.strip_suffix(LORA_B_BIAS_SUFFIX) {
            let module_name = prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .context("LoRA B-bias tensor lost validated PEFT prefix")?;
            validate_adapter_pair_target(module_name, config)?;
            if b_bias_keys
                .insert(prefix.to_string(), key.to_string())
                .is_some()
            {
                bail!("duplicate LoRA B-bias tensor for module {module_name:?}");
            }
        } else if let Some(prefix) = key
            .strip_suffix(LORA_MAGNITUDE_WEIGHT_SUFFIX)
            .or_else(|| key.strip_suffix(LORA_MAGNITUDE_SUFFIX))
        {
            let module_name = prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .context("DoRA magnitude tensor lost validated PEFT prefix")?;
            validate_adapter_pair_target(module_name, config)?;
            if magnitude_keys
                .insert(prefix.to_string(), key.to_string())
                .is_some()
            {
                bail!("duplicate DoRA magnitude tensor for module {module_name:?}");
            }
        } else {
            let canonical = key
                .strip_prefix(PEFT_STATE_PREFIX)
                .context("adapter replacement tensor lost validated PEFT prefix")?;
            if let Some(module_name) = normalize_trainable_tokens_tensor(canonical) {
                if config.trainable_token_indices.is_none()
                    || matches!(
                        config.trainable_token_indices,
                        Some(serde_json::Value::Null)
                    )
                {
                    bail!(
                        "adapter contains trainable-token tensor {key:?} but adapter_config.json has no trainable_token_indices"
                    );
                }
                let _ = decode_float_tensor(key, &tensor)?;
                trainable_token_replacements.push(TrainableTokenReplacement {
                    adapter_key: key.to_string(),
                    module_name,
                });
                continue;
            }
            let target_name = normalize_modules_to_save_tensor(canonical, modules_to_save)
                .or_else(|| {
                    tensor_belongs_to_modules_to_save(canonical, modules_to_save)
                        .then_some(canonical.to_owned())
                })
                .or_else(|| {
                    let bias_mode = if config.bias.is_empty() {
                        "none"
                    } else {
                        config.bias.as_str()
                    };
                    let module_name = canonical.strip_suffix(".bias")?;
                    if bias_mode.eq_ignore_ascii_case("all")
                        || (bias_mode.eq_ignore_ascii_case("lora_only")
                            && target_module_declared(module_name, config))
                    {
                        Some(canonical.to_owned())
                    } else {
                        None
                    }
                })
                .with_context(|| {
                    format!(
                        "unsupported tensor {key:?} in adapter; expected standard LoRA A/B matrices, configured modules_to_save tensors, or PEFT bias tensors allowed by bias={:?}",
                        config.bias
                    )
                })?;
            // Decode now for dtype/non-finite validation even though replacement
            // is materialized after pair discovery.
            let _ = decode_float_tensor(key, &tensor)?;
            replacements.push(AdapterReplacement {
                adapter_key: key.to_string(),
                target_name,
            });
        }
    }

    if a_keys.is_empty() || a_keys.keys().ne(b_keys.keys()) {
        let missing_b = a_keys
            .keys()
            .filter(|key| !b_keys.contains_key(*key))
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        let missing_a = b_keys
            .keys()
            .filter(|key| !a_keys.contains_key(*key))
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        bail!(
            "PEFT adapter has incomplete LoRA A/B pairs; missing_B={missing_b:?} missing_A={missing_a:?}"
        );
    }
    for prefix in b_bias_keys.keys() {
        if !a_keys.contains_key(prefix) {
            bail!("LoRA B-bias tensor has no matching LoRA A/B pair: {prefix:?}");
        }
    }
    for prefix in magnitude_keys.keys() {
        if !a_keys.contains_key(prefix) {
            bail!("DoRA magnitude tensor has no matching LoRA A/B pair: {prefix:?}");
        }
    }
    if !config.lora_bias && !b_bias_keys.is_empty() {
        bail!("adapter contains LoRA B-bias tensors but adapter_config.json has lora_bias=false");
    }
    if !config.use_dora && !magnitude_keys.is_empty() {
        bail!("adapter contains DoRA magnitude tensors but adapter_config.json has use_dora=false");
    }
    if config.lora_bias && b_bias_keys.len() != a_keys.len() {
        bail!("PEFT lora_bias=true requires one saved LoRA B bias for every adapter pair");
    }
    if config.use_dora && magnitude_keys.len() != a_keys.len() {
        bail!("PEFT use_dora=true requires one magnitude vector for every adapter pair");
    }
    let pairs = a_keys
        .into_iter()
        .map(|(prefix, a_key)| {
            let raw_module_name = prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .expect("prefix validated above")
                .to_string();
            let nested_parameter_wrapper =
                raw_module_name.ends_with(PEFT_PARAMETER_BASE_LAYER_SUFFIX);
            let module_name = raw_module_name
                .strip_suffix(PEFT_PARAMETER_BASE_LAYER_SUFFIX)
                .unwrap_or(&raw_module_name)
                .to_string();
            AdapterPair {
                module_name,
                nested_parameter_wrapper,
                a_key,
                b_key: b_keys
                    .get(&prefix)
                    .expect("A/B key sets validated above")
                    .clone(),
                b_bias_key: b_bias_keys.get(&prefix).cloned(),
                magnitude_key: magnitude_keys.get(&prefix).cloned(),
            }
        })
        .collect::<Vec<_>>();
    Ok((pairs, replacements, trainable_token_replacements))
}

fn normalize_trainable_tokens_tensor(tensor_name: &str) -> Option<String> {
    const SUFFIX: &str = ".token_adapter.trainable_tokens_delta";
    const DEFAULT_SUFFIX: &str = ".token_adapter.trainable_tokens_delta.default";
    let module_name = tensor_name
        .strip_suffix(DEFAULT_SUFFIX)
        .or_else(|| tensor_name.strip_suffix(SUFFIX))?;
    (!module_name.is_empty()).then(|| module_name.to_owned())
}

fn module_selector_is_empty(selector: &ModuleSelector) -> bool {
    match selector {
        ModuleSelector::Regex(pattern) => pattern.trim().is_empty(),
        ModuleSelector::Modules(modules) => modules.is_empty(),
    }
}

fn validate_module_selector(name: &str, selector: &ModuleSelector) -> Result<()> {
    match selector {
        ModuleSelector::Regex(pattern) => {
            if pattern.trim().is_empty() {
                bail!("PEFT {name} regex must not be empty");
            }
            Regex::new(pattern)
                .with_context(|| format!("PEFT {name} contains an invalid regex"))?;
        }
        ModuleSelector::Modules(modules) => {
            if modules.iter().any(|module| module.trim().is_empty()) {
                bail!("PEFT {name} entries must not be empty");
            }
            if modules.len()
                != modules
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    .len()
            {
                bail!("PEFT {name} contains duplicate module names");
            }
        }
    }
    Ok(())
}

fn selector_matches_module(module_name: &str, selector: &ModuleSelector) -> bool {
    match selector {
        ModuleSelector::Regex(pattern) => Regex::new(pattern).is_ok_and(|regex| {
            regex
                .find(module_name)
                .is_some_and(|matched| matched.start() == 0 && matched.end() == module_name.len())
        }),
        ModuleSelector::Modules(modules) => modules
            .iter()
            .any(|target| module_name == target || module_name.ends_with(&format!(".{target}"))),
    }
}

fn target_module_declared(module_name: &str, config: &LoraAdapterConfig) -> bool {
    selector_matches_module(module_name, &config.target_modules)
        && !selector_matches_module(module_name, &config.exclude_modules)
}

fn tensor_belongs_to_modules_to_save(tensor_name: &str, modules_to_save: &[String]) -> bool {
    modules_to_save.iter().any(|module| {
        tensor_name
            .strip_prefix(module)
            .is_some_and(|suffix| suffix.starts_with('.'))
            || tensor_name.contains(&format!(".{module}."))
    })
}

fn normalize_modules_to_save_tensor(
    tensor_name: &str,
    modules_to_save: &[String],
) -> Option<String> {
    const WRAPPER: &str = ".modules_to_save.";
    let wrapper_index = tensor_name.find(WRAPPER)?;
    let module_name = &tensor_name[..wrapper_index];
    if !modules_to_save
        .iter()
        .any(|module| module_name == module || module_name.ends_with(&format!(".{module}")))
    {
        return None;
    }
    let wrapped = &tensor_name[wrapper_index + WRAPPER.len()..];
    let parameter = wrapped
        .strip_prefix("default.")
        .unwrap_or(wrapped)
        .trim_start_matches('.');
    if parameter.is_empty() {
        return None;
    }
    Some(format!("{module_name}.{parameter}"))
}

fn validate_target_module(module_name: &str, config: &LoraAdapterConfig) -> Result<()> {
    if target_module_declared(module_name, config) {
        return Ok(());
    }
    bail!("LoRA tensor module {module_name:?} is not declared by adapter target_modules")
}

fn validate_adapter_pair_target(module_name: &str, config: &LoraAdapterConfig) -> Result<()> {
    if target_module_declared(module_name, config)
        || config
            .target_parameters
            .as_ref()
            .is_some_and(|parameters| !parameters.is_empty())
    {
        return Ok(());
    }
    validate_target_module(module_name, config)
}

fn pattern_value<'a, T>(module_name: &str, pattern: &'a BTreeMap<String, T>) -> Option<&'a T> {
    pattern
        .iter()
        .filter(|(name, _)| {
            module_name == name.as_str() || module_name.ends_with(&format!(".{name}"))
        })
        .max_by_key(|(name, _)| name.len())
        .map(|(_, value)| value)
}

fn resolve_target_parameter(
    pair: &AdapterPair,
    config: &LoraAdapterConfig,
    base_tensors: &SafeTensors<'_>,
) -> Result<Option<String>> {
    let linear_target = format!("{}.weight", pair.module_name);
    if !pair.nested_parameter_wrapper
        && target_module_declared(&pair.module_name, config)
        && base_tensors.tensor(&linear_target).is_ok()
    {
        return Ok(None);
    }

    let Some(target_parameters) = config.target_parameters.as_deref() else {
        if pair.nested_parameter_wrapper {
            bail!(
                "adapter pair {:?} uses a nested PEFT ParamWrapper but adapter_config.json has no target_parameters",
                pair.module_name
            );
        }
        return Ok(None);
    };
    if target_parameters.is_empty() {
        return Ok(None);
    }

    let prefix = format!("{}.", pair.module_name);
    let names = base_tensors.names();
    let mut candidates = Vec::<String>::new();
    for declared in target_parameters {
        let matches = names
            .iter()
            .filter(|name| {
                name.starts_with(&prefix)
                    && (*name == declared || name.ends_with(&format!(".{declared}")))
            })
            .copied()
            .collect::<Vec<_>>();
        if matches.len() > 1 {
            bail!(
                "PEFT target_parameters entry {declared:?} ambiguously resolves beneath {:?}: {matches:?}",
                pair.module_name
            );
        }
        if let Some(name) = matches.first() {
            if !candidates.iter().any(|candidate| candidate == *name) {
                candidates.push((*name).to_owned());
            }
        }
    }
    if candidates.is_empty() {
        bail!(
            "adapter pair {:?} does not resolve to a base tensor declared by target_parameters={target_parameters:?}",
            pair.module_name
        );
    }
    if candidates.len() > 2 {
        bail!(
            "native PEFT ParamWrapper merge currently supports at most two target_parameters on one module; {:?} resolves to {candidates:?}",
            pair.module_name
        );
    }
    let selected = match candidates.as_slice() {
        [only] => only,
        [inner, _outer] if pair.nested_parameter_wrapper => inner,
        [_inner, outer] => outer,
        _ => unreachable!("candidate cardinality validated above"),
    };
    Ok(Some(selected.clone()))
}

fn lora_scale(alpha: f64, rank: usize, use_rslora: bool, target: &str) -> Result<f32> {
    if rank == 0 {
        bail!("LoRA target {target:?} has zero rank");
    }
    let scale = if use_rslora {
        alpha / (rank as f64).sqrt()
    } else {
        alpha / rank as f64
    } as f32;
    if !scale.is_finite() {
        bail!("LoRA target {target:?} produced non-finite scaling");
    }
    Ok(scale)
}

fn add_packed_parameter_lora_delta(
    base: &mut [f32],
    a: &[f32],
    b: &[f32],
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    scale: f32,
) -> Result<()> {
    let expanded_rank = rank
        .checked_mul(num_experts)
        .context("packed target_parameters rank overflow")?;
    let base_len = num_experts
        .checked_mul(in_dim)
        .and_then(|value| value.checked_mul(out_dim))
        .context("packed target_parameters base size overflow")?;
    let a_len = expanded_rank
        .checked_mul(in_dim)
        .context("packed target_parameters A size overflow")?;
    let b_len = out_dim
        .checked_mul(expanded_rank)
        .context("packed target_parameters B size overflow")?;
    if base.len() != base_len || a.len() != a_len || b.len() != b_len {
        bail!(
            "packed target_parameters payload lengths do not match geometry: base={} A={} B={} expected base={base_len} A={a_len} B={b_len}",
            base.len(),
            a.len(),
            b.len()
        );
    }

    // PEFT ParamWrapper stores a 3-D base Parameter as [expert, in, out],
    // LoRA-A as [expert * rank, in], and LoRA-B as [out, rank * expert].
    // This is the flattened form of PEFT 0.18's einsum
    // `o r e, e r i -> e i o`.
    for expert in 0..num_experts {
        for input in 0..in_dim {
            for output in 0..out_dim {
                let mut sum = 0.0f32;
                for adapter_rank in 0..rank {
                    let a_index = (expert * rank + adapter_rank) * in_dim + input;
                    let b_index = output * expanded_rank + adapter_rank * num_experts + expert;
                    sum = b[b_index].mul_add(a[a_index], sum);
                }
                let base_index = (expert * in_dim + input) * out_dim + output;
                base[base_index] = scale.mul_add(sum, base[base_index]);
                if !base[base_index].is_finite() {
                    bail!("packed target_parameters merge produced a non-finite model weight");
                }
            }
        }
    }
    Ok(())
}

fn apply_dora_magnitude(
    weight: &mut [f32],
    magnitude: &[f32],
    in_dim: usize,
    out_dim: usize,
    transpose: bool,
) -> Result<()> {
    if weight.len() != in_dim * out_dim || magnitude.len() != out_dim {
        bail!("DoRA merge tensor geometry mismatch");
    }
    for output in 0..out_dim {
        let mut sum_sq = 0.0f32;
        for input in 0..in_dim {
            let index = if transpose {
                input * out_dim + output
            } else {
                output * in_dim + input
            };
            sum_sq = weight[index].mul_add(weight[index], sum_sq);
        }
        let norm = sum_sq.sqrt().max(1.0e-30);
        let factor = magnitude[output] / norm;
        if !factor.is_finite() {
            bail!("DoRA merge produced a non-finite magnitude factor");
        }
        for input in 0..in_dim {
            let index = if transpose {
                input * out_dim + output
            } else {
                output * in_dim + input
            };
            weight[index] *= factor;
            if !weight[index].is_finite() {
                bail!("DoRA merge produced a non-finite model weight");
            }
        }
    }
    Ok(())
}

fn add_lora_delta(
    base: &mut [f32],
    a: &[f32],
    b: &[f32],
    out_dim: usize,
    rank: usize,
    in_dim: usize,
    scale: f32,
    transpose: bool,
) -> Result<()> {
    let a_len = rank.checked_mul(in_dim).context("LoRA A size overflow")?;
    let b_len = out_dim.checked_mul(rank).context("LoRA B size overflow")?;
    let base_len = out_dim
        .checked_mul(in_dim)
        .context("LoRA merged tensor size overflow")?;
    if a.len() != a_len || b.len() != b_len || base.len() != base_len {
        bail!(
            "LoRA matrix payload lengths do not match geometry: base={} A={} B={} expected base={base_len} A={a_len} B={b_len}",
            base.len(),
            a.len(),
            b.len()
        );
    }
    for out in 0..out_dim {
        for input in 0..in_dim {
            let mut sum = 0.0f32;
            for inner in 0..rank {
                sum += b[out * rank + inner] * a[inner * in_dim + input];
            }
            let index = if transpose {
                input * out_dim + out
            } else {
                out * in_dim + input
            };
            base[index] += sum * scale;
            if !base[index].is_finite() {
                bail!("LoRA merge produced a non-finite model weight");
            }
        }
    }
    Ok(())
}

fn decode_float_tensor(name: &str, tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    let raw = tensor.data();
    let mut values = Vec::with_capacity(match tensor.dtype() {
        Dtype::F32 => raw.len() / 4,
        Dtype::F16 | Dtype::BF16 => raw.len() / 2,
        dtype => bail!("tensor {name:?} must be F32/F16/BF16, got {dtype:?}"),
    });
    match tensor.dtype() {
        Dtype::F32 => {
            if raw.len() % 4 != 0 {
                bail!("tensor {name:?} has invalid FP32 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Dtype::F16 => {
            if raw.len() % 2 != 0 {
                bail!("tensor {name:?} has invalid FP16 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(2) {
                values.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        Dtype::BF16 => {
            if raw.len() % 2 != 0 {
                bail!("tensor {name:?} has invalid BF16 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(2) {
                values.push(f32::from_bits(
                    u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16,
                ));
            }
        }
        _ => unreachable!("dtype validated above"),
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("tensor {name:?} contains non-finite floating-point values");
    }
    Ok(values)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let value = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut mantissa = u32::from(fraction);
                let mut shift = 0u32;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    shift += 1;
                }
                mantissa &= 0x03ff;
                let exp = 127u32 - 15 - shift + 1;
                sign | (exp << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => {
            let exp = u32::from(exponent) + (127 - 15);
            sign | (exp << 23) | (u32::from(fraction) << 13)
        }
    };
    f32::from_bits(value)
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use safetensors::{serialize_to_file, tensor::TensorView};
    use serde_json::json;

    use super::*;

    #[test]
    fn standalone_lora_config_accepts_peft_regex_module_selectors() -> Result<()> {
        let config: LoraAdapterConfig = serde_json::from_value(json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": "model\\.layers\\.0\\.(q_proj|v_proj)",
            "exclude_modules": "model\\.layers\\.0\\.v_proj",
            "bias": "none"
        }))?;

        validate_adapter_config(&config)?;
        assert!(target_module_declared("model.layers.0.q_proj", &config));
        assert!(!target_module_declared("model.layers.0.v_proj", &config));
        // Scalar PEFT selectors use re.fullmatch semantics rather than a
        // substring search.
        assert!(!target_module_declared(
            "base.model.layers.0.q_proj.extra",
            &config
        ));
        Ok(())
    }

    #[test]
    fn standalone_lora_config_rejects_topology_mutating_layer_replication() -> Result<()> {
        let config: LoraAdapterConfig = serde_json::from_value(json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "layer_replication": [[0, 2], [1, 3]],
            "bias": "none"
        }))?;

        let error = validate_adapter_config(&config)
            .expect_err("standalone merging cannot preserve replicated topology");
        assert!(error
            .to_string()
            .contains("changes the base-model topology"));
        Ok(())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hierarchos-native-lora-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn write_tensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Result<()> {
        let bytes = tensors
            .iter()
            .map(|(_, _, values)| f32_bytes(values))
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .zip(&bytes)
            .map(|((name, shape, _), bytes)| {
                Ok((*name, TensorView::new(Dtype::F32, shape.clone(), bytes)?))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, None, path)?;
        Ok(())
    }

    fn write_bound_manifest(base: &Path, adapter_dir: &Path, arch_hash: &str) -> Result<()> {
        let config = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME);
        let weights = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME);
        let manifest_path = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME);
        let manifest = json!({
            "manifest_version": 1,
            "format": HIERARCHOS_LORA_ADAPTER_MANIFEST_FORMAT,
            "base_checkpoint": {
                "filename": base.file_name().unwrap().to_string_lossy(),
                "sha256": sha256_file(base)?,
            },
            "architecture_contract_sha256": arch_hash,
            "adapter_files": {
                HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME: sha256_file(&config)?,
                HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME: sha256_file(&weights)?,
            },
            "finetune_run_identity": {},
            "tokenizer_identity": {},
            "lora_geometry": {},
        });
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        let checksum = sha256_file(&manifest_path)?;
        fs::write(
            format!("{}.sha256", manifest_path.display()),
            format!("{checksum}  {HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME}\n"),
        )?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_applies_b_times_a_and_saved_ltm() -> Result<()> {
        let root = temp_dir("merge");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "1".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                (
                    "block.key.weight",
                    vec![2, 3],
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ),
                ("ltm.key_proj.weight", vec![1, 2], vec![7.0, 8.0]),
                ("untouched.weight", vec![1], vec![9.0]),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.05,
                "target_modules": ["key"],
                "modules_to_save": ["ltm"],
                "bias": "none",
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.block.key.lora_A.weight",
                    vec![2, 3],
                    vec![1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
                ),
                (
                    "base_model.model.block.key.lora_B.weight",
                    vec![2, 2],
                    vec![1.0, 2.0, 3.0, 4.0],
                ),
                (
                    "base_model.model.ltm.key_proj.weight",
                    vec![1, 2],
                    vec![10.0, 11.0],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 1);
        assert_eq!(report.replaced_module_tensors, 1);
        let (_, merged) = crate::checkpoint::read_f32_tensor(&output, "block.key.weight")?;
        // B@A = [[1,2,4],[3,4,10]], scale=alpha/r=2.
        assert_eq!(merged, vec![3.0, 6.0, 11.0, 10.0, 13.0, 26.0]);
        let (_, ltm) = crate::checkpoint::read_f32_tensor(&output, "ltm.key_proj.weight")?;
        assert_eq!(ltm, vec![10.0, 11.0]);
        let (_, untouched) = crate::checkpoint::read_f32_tensor(&output, "untouched.weight")?;
        assert_eq!(untouched, vec![9.0]);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_rejects_unbound_base() -> Result<()> {
        let root = temp_dir("binding");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "2".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(&base, &[("key.weight", vec![1, 1], vec![1.0])])?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec(&json!({
                "peft_type":"LORA", "task_type":"CAUSAL_LM", "r":1,
                "lora_alpha":1, "target_modules":["key"], "bias":"none"
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                ("base_model.model.key.lora_A.weight", vec![1, 1], vec![1.0]),
                ("base_model.model.key.lora_B.weight", vec![1, 1], vec![1.0]),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;
        // Mutate the base after the adapter has been cryptographically bound.
        write_tensors(&base, &[("key.weight", vec![1, 1], vec![2.0])])?;
        let error = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)
            .expect_err("base binding mismatch must fail");
        assert!(error.to_string().contains("not bound to this exact base"));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_supports_bias_all_and_generic_modules_to_save() -> Result<()> {
        let root = temp_dir("bias-all-modules-to-save");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "3".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                ("block.key.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
                ("block.key.bias", vec![2], vec![0.1, 0.2]),
                ("block.norm.bias", vec![2], vec![0.3, 0.4]),
                ("head.weight", vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]),
                ("head.bias", vec![2], vec![0.5, 0.6]),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 2,
                "target_modules": ["key"],
                "modules_to_save": ["head"],
                "bias": "all"
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.block.key.lora_A.weight",
                    vec![1, 2],
                    vec![1.0, 2.0],
                ),
                (
                    "base_model.model.block.key.lora_B.weight",
                    vec![2, 1],
                    vec![3.0, 4.0],
                ),
                ("base_model.model.block.key.bias", vec![2], vec![1.1, 1.2]),
                ("base_model.model.block.norm.bias", vec![2], vec![1.3, 1.4]),
                (
                    "base_model.model.head.modules_to_save.default.weight",
                    vec![2, 2],
                    vec![9.0, 10.0, 11.0, 12.0],
                ),
                (
                    "base_model.model.head.modules_to_save.default.bias",
                    vec![2],
                    vec![1.5, 1.6],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 1);
        assert_eq!(report.replaced_module_tensors, 4);
        let (_, merged) = crate::checkpoint::read_f32_tensor(&output, "block.key.weight")?;
        // B@A = [[3, 6], [4, 8]], scale=alpha/r=2.
        assert_eq!(merged, vec![7.0, 14.0, 11.0, 20.0]);
        let (_, target_bias) = crate::checkpoint::read_f32_tensor(&output, "block.key.bias")?;
        assert_eq!(target_bias, vec![1.1, 1.2]);
        let (_, norm_bias) = crate::checkpoint::read_f32_tensor(&output, "block.norm.bias")?;
        assert_eq!(norm_bias, vec![1.3, 1.4]);
        let (_, head_weight) = crate::checkpoint::read_f32_tensor(&output, "head.weight")?;
        assert_eq!(head_weight, vec![9.0, 10.0, 11.0, 12.0]);
        let (_, head_bias) = crate::checkpoint::read_f32_tensor(&output, "head.bias")?;
        assert_eq!(head_bias, vec![1.5, 1.6]);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_supports_lora_only_bias_and_null_modules_to_save() -> Result<()> {
        let root = temp_dir("lora-only-bias");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "4".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                ("block.key.weight", vec![1, 1], vec![2.0]),
                ("block.key.bias", vec![1], vec![0.0]),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["key"],
                "modules_to_save": null,
                "bias": "lora_only"
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.block.key.lora_A.weight",
                    vec![1, 1],
                    vec![2.0],
                ),
                (
                    "base_model.model.block.key.lora_B.weight",
                    vec![1, 1],
                    vec![3.0],
                ),
                ("base_model.model.block.key.bias", vec![1], vec![0.75]),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 1);
        assert_eq!(report.replaced_module_tensors, 1);
        let (_, merged) = crate::checkpoint::read_f32_tensor(&output, "block.key.weight")?;
        assert_eq!(merged, vec![8.0]);
        let (_, bias) = crate::checkpoint::read_f32_tensor(&output, "block.key.bias")?;
        assert_eq!(bias, vec![0.75]);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_supports_trainable_token_indices_and_corda_metadata() -> Result<()> {
        let root = temp_dir("trainable-tokens-list");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "8".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                ("block.key.weight", vec![1, 1], vec![2.0]),
                (
                    "model.embed_tokens.weight",
                    vec![5, 2],
                    vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1],
                ),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["key"],
                "bias": "none",
                "trainable_token_indices": [1, 3],
                "init_lora_weights": "corda",
                "corda_config": {"corda_method": "ipm"}
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.block.key.lora_A.weight",
                    vec![1, 1],
                    vec![2.0],
                ),
                (
                    "base_model.model.block.key.lora_B.weight",
                    vec![1, 1],
                    vec![3.0],
                ),
                (
                    "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta",
                    vec![2, 2],
                    vec![10.0, 11.0, 30.0, 31.0],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 1);
        assert_eq!(report.replaced_module_tensors, 1);
        let (_, merged) = crate::checkpoint::read_f32_tensor(&output, "block.key.weight")?;
        assert_eq!(merged, vec![8.0]);
        let (_, embedding) =
            crate::checkpoint::read_f32_tensor(&output, "model.embed_tokens.weight")?;
        assert_eq!(
            embedding,
            vec![0.0, 0.1, 10.0, 11.0, 2.0, 2.1, 30.0, 31.0, 4.0, 4.1]
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_supports_trainable_token_module_map_and_default_key() -> Result<()> {
        let root = temp_dir("trainable-tokens-map");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "9".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                ("key.weight", vec![1, 1], vec![1.0]),
                (
                    "transformer.wte.weight",
                    vec![4, 3],
                    vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
                ),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["key"],
                "bias": "none",
                "trainable_token_indices": {"wte": [0, 2]}
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                ("base_model.model.key.lora_A.weight", vec![1, 1], vec![1.0]),
                ("base_model.model.key.lora_B.weight", vec![1, 1], vec![2.0]),
                (
                    "base_model.model.transformer.wte.token_adapter.trainable_tokens_delta.default",
                    vec![2, 3],
                    vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.replaced_module_tensors, 1);
        let (_, embedding) = crate::checkpoint::read_f32_tensor(&output, "transformer.wte.weight")?;
        assert_eq!(
            embedding,
            vec![10.0, 11.0, 12.0, 1.0, 1.1, 1.2, 20.0, 21.0, 22.0, 3.0, 3.1, 3.2,]
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_supports_lora_b_bias() -> Result<()> {
        let root = temp_dir("lora-b-bias");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "6".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                ("block.key.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
                ("block.key.bias", vec![2], vec![1.0, 2.0]),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 2,
                "target_modules": ["key"],
                "bias": "none",
                "lora_bias": true
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.block.key.lora_A.weight",
                    vec![1, 2],
                    vec![1.0, 0.0],
                ),
                (
                    "base_model.model.block.key.lora_B.weight",
                    vec![2, 1],
                    vec![3.0, 4.0],
                ),
                (
                    "base_model.model.block.key.lora_B.bias",
                    vec![2],
                    vec![0.5, 1.0],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 1);
        let (_, weight) = crate::checkpoint::read_f32_tensor(&output, "block.key.weight")?;
        assert_eq!(weight, vec![7.0, 2.0, 11.0, 4.0]);
        let (_, bias) = crate::checkpoint::read_f32_tensor(&output, "block.key.bias")?;
        assert_eq!(bias, vec![2.0, 4.0]);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_supports_dora_magnitude_vectors() -> Result<()> {
        let root = temp_dir("dora");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "7".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[("block.key.weight", vec![2, 2], vec![3.0, 0.0, 0.0, 4.0])],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["key"],
                "bias": "none",
                "use_dora": true
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.block.key.lora_A.weight",
                    vec![1, 2],
                    vec![1.0, 0.0],
                ),
                (
                    "base_model.model.block.key.lora_B.weight",
                    vec![2, 1],
                    vec![1.0, 0.0],
                ),
                (
                    "base_model.model.block.key.lora_magnitude_vector",
                    vec![2],
                    vec![8.0, 2.0],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 1);
        let (_, weight) = crate::checkpoint::read_f32_tensor(&output, "block.key.weight")?;
        assert_eq!(weight, vec![8.0, 0.0, 0.0, 2.0]);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_supports_packed_target_parameters_and_current_task_types() -> Result<()> {
        let root = temp_dir("packed-target-parameters");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "5".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                (
                    "model.layers.0.feed_forward.experts.gate_up_proj",
                    vec![2, 2, 3],
                    vec![0.0; 12],
                ),
                (
                    "model.layers.0.feed_forward.experts.down_proj",
                    vec![2, 3, 2],
                    vec![0.0; 12],
                ),
                ("untouched.weight", vec![1], vec![9.0]),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "FEATURE_EXTRACTION",
                "r": 1,
                "lora_alpha": 2,
                "target_modules": null,
                "target_parameters": ["gate_up_proj", "down_proj"],
                "bias": "none"
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.model.layers.0.feed_forward.experts.base_layer.lora_A.weight",
                    vec![2, 2],
                    vec![1.0, 2.0, 3.0, 4.0],
                ),
                (
                    "base_model.model.model.layers.0.feed_forward.experts.base_layer.lora_B.weight",
                    vec![3, 2],
                    vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ),
                (
                    "base_model.model.model.layers.0.feed_forward.experts.lora_A.weight",
                    vec![2, 3],
                    vec![1.0, 0.0, 2.0, 0.0, 3.0, 1.0],
                ),
                (
                    "base_model.model.model.layers.0.feed_forward.experts.lora_B.weight",
                    vec![2, 2],
                    vec![4.0, 5.0, 6.0, 7.0],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 2);
        assert_eq!(report.replaced_module_tensors, 0);
        let (_, gate_up) = crate::checkpoint::read_f32_tensor(
            &output,
            "model.layers.0.feed_forward.experts.gate_up_proj",
        )?;
        assert_eq!(
            gate_up,
            vec![10.0, 14.0, 18.0, 20.0, 28.0, 36.0, 36.0, 48.0, 60.0, 48.0, 64.0, 80.0]
        );
        let (_, down) = crate::checkpoint::read_f32_tensor(
            &output,
            "model.layers.0.feed_forward.experts.down_proj",
        )?;
        assert_eq!(
            down,
            vec![8.0, 12.0, 0.0, 0.0, 16.0, 24.0, 0.0, 0.0, 30.0, 42.0, 10.0, 14.0]
        );
        let (_, untouched) = crate::checkpoint::read_f32_tensor(&output, "untouched.weight")?;
        assert_eq!(untouched, vec![9.0]);

        fs::remove_dir_all(root)?;
        Ok(())
    }
}
