//! Standard HF PEFT-LoRA adapters for the dedicated V4.1 text graph.
//!
//! The released package stays exactly as published: packed FP8/FP4 blocks are
//! never rewritten, so a merged file cannot silently change the base bits. An
//! adapter is an explicit additive delta at the projection boundary,
//!
//! ```text
//! y = base(x) + scale * B(A(x)),   scale = alpha / r  (alpha / sqrt(r) for rsLoRA)
//! ```
//!
//! evaluated in FP32 from the module input, added to the base result in FP32,
//! and rounded to the stream dtype exactly once. That is the same arithmetic
//! PEFT performs on a non-quantized `nn.Linear`; only the base GEMM keeps the
//! released packed contract. `modules_to_save`, DoRA, QALoRA, trainable
//! tokens, layer replication and adapter-bias replacement remain fail-closed
//! until their own <=2e-7 gate exists for this graph.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const DEEPSEEK_V41_LORA_CONFIG: &str = "adapter_config.json";
pub const DEEPSEEK_V41_LORA_WEIGHTS: &str = "adapter_model.safetensors";
const PEFT_STATE_PREFIX: &str = "base_model.model.";
const LORA_A_SUFFIX: &str = ".lora_A.weight";
const LORA_B_SUFFIX: &str = ".lora_B.weight";

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TargetSelector {
    All(String),
    Names(Vec<String>),
}

impl Default for TargetSelector {
    fn default() -> Self {
        Self::Names(Vec::new())
    }
}

#[derive(Debug, Deserialize)]
struct AdapterConfig {
    #[serde(default)]
    peft_type: String,
    #[serde(default)]
    task_type: String,
    #[serde(default)]
    r: usize,
    #[serde(default)]
    lora_alpha: f64,
    #[serde(default)]
    lora_dropout: f64,
    #[serde(default)]
    target_modules: TargetSelector,
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_qalora: bool,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    bias: String,
    #[serde(default)]
    lora_bias: bool,
    #[serde(default)]
    rank_pattern: BTreeMap<String, usize>,
    #[serde(default)]
    alpha_pattern: BTreeMap<String, f64>,
    #[serde(default)]
    target_parameters: Option<Vec<String>>,
    #[serde(default)]
    trainable_token_indices: Option<Value>,
    #[serde(default)]
    alora_invocation_tokens: Option<Value>,
    #[serde(default)]
    arrow_config: Option<Value>,
    #[serde(default)]
    layer_replication: Option<Value>,
}

/// One adapter projection, already resolved to the checkpoint's canonical
/// tensor name (for example `layers.3.attn.wq_b.weight`).
#[derive(Debug, Clone)]
pub struct DeepseekV41LoraModule {
    pub target: String,
    pub rank: usize,
    pub alpha: f64,
    pub scale: f32,
    /// `[rank, in_features]`, FP32, row-major.
    pub a: Arc<[f32]>,
    /// `[out_features, rank]`, FP32, row-major.
    pub b: Arc<[f32]>,
}

/// A validated, read-only adapter. Every tensor is decoded once and shared by
/// the resident, streamed and test paths.
#[derive(Debug, Clone)]
pub struct DeepseekV41LoraAdapter {
    pub name: String,
    pub source_sha256: String,
    modules: BTreeMap<String, DeepseekV41LoraModule>,
    /// Preserve PEFT metadata, including per-module ranks, rsLoRA and fields
    /// unknown to this runtime, through an unmerged adapter export.
    config: Value,
}

impl DeepseekV41LoraAdapter {
    /// Initialize standard, zero-dropout additive factors against actual
    /// checkpoint shapes. An empty target list means every supported linear
    /// boundary in backbone layers; expert targets remain independent modules.
    pub fn initialize(
        checkpoint: &mut super::DeepseekV41Checkpoint,
        targets: &[String],
        rank: usize,
        alpha: f64,
        rslora: bool,
        seed: u64,
    ) -> Result<Self> {
        if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
            bail!("V4.1 LoRA initialization requires positive rank and alpha");
        }
        let mut modules = BTreeMap::new();
        let mut state = seed;
        for tensor in checkpoint.tensor_names()? {
            let Some(target) = tensor.strip_suffix(".weight") else {
                continue;
            };
            if !super::checkpoint::lora_target_supported(target) {
                continue;
            }
            let layer = target
                .split('.')
                .nth(1)
                .and_then(|n| n.parse::<usize>().ok());
            if layer.is_none_or(|n| n >= checkpoint.config.text.num_hidden_layers) {
                continue;
            }
            if !targets.is_empty()
                && !targets
                    .iter()
                    .any(|t| target == t || target.ends_with(&format!(".{t}")))
            {
                continue;
            }
            let shape = checkpoint.tensor_shape(&tensor)?;
            if shape.len() != 2 || shape.contains(&0) {
                bail!("V4.1 adapter target {target} is not a matrix");
            }
            let a_len = rank
                .checked_mul(shape[1])
                .context("V4.1 LoRA A size overflow")?;
            let b_len = rank
                .checked_mul(shape[0])
                .context("V4.1 LoRA B size overflow")?;
            let bound = (shape[1] as f64).sqrt().recip() as f32;
            let a = (0..a_len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((state >> 40) as f32 / 16777216.0) * 2.0 - 1.0) * bound
                })
                .collect::<Vec<_>>();
            let scale = (alpha
                / if rslora {
                    (rank as f64).sqrt()
                } else {
                    rank as f64
                }) as f32;
            modules.insert(
                target.to_owned(),
                DeepseekV41LoraModule {
                    target: target.to_owned(),
                    rank,
                    alpha,
                    scale,
                    a: a.into(),
                    b: vec![0.0; b_len].into(),
                },
            );
        }
        if modules.is_empty() {
            bail!("V4.1 LoRA initialization found no matching supported targets");
        }
        let config = serde_json::json!({"peft_type":"LORA","task_type":"CAUSAL_LM","r":rank,"lora_alpha":alpha,
            "lora_dropout":0.0,"bias":"none","use_rslora":rslora,"target_modules":modules.keys().collect::<Vec<_>>()});
        let source_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&config)?));
        Ok(Self {
            name: "initialized".to_owned(),
            source_sha256,
            modules,
            config,
        })
    }

    pub fn dropout(&self) -> f64 {
        self.config
            .get("lora_dropout")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    }
    /// Load and validate a standard PEFT `adapter_config.json` +
    /// `adapter_model.safetensors` directory. Nothing is mutated on disk.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        if !directory.is_dir() {
            bail!(
                "V4.1 LoRA adapter directory does not exist: {}",
                directory.display()
            );
        }
        reject_python_adapter_files(directory)?;
        let config_path = directory.join(DEEPSEEK_V41_LORA_CONFIG);
        let weights_path = directory.join(DEEPSEEK_V41_LORA_WEIGHTS);
        for path in [&config_path, &weights_path] {
            if !path.is_file() {
                bail!("V4.1 LoRA adapter is missing {}", path.display());
            }
        }
        let config_bytes =
            fs::read(&config_path).with_context(|| format!("reading {}", config_path.display()))?;
        let config_value: Value = serde_json::from_slice(&config_bytes)?;
        let config: AdapterConfig = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("decoding {}", config_path.display()))?;
        Self::validate_config(&config)?;
        let bytes = fs::read(&weights_path)
            .with_context(|| format!("reading {}", weights_path.display()))?;
        let tensors = safetensors::tensor::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parsing {}", weights_path.display()))?;
        let mut pairs = BTreeMap::<String, (Option<String>, Option<String>)>::new();
        for key in tensors.names() {
            let Some(rest) = key.strip_prefix(PEFT_STATE_PREFIX) else {
                bail!("unsupported V4.1 adapter tensor key {key:?}; expected {PEFT_STATE_PREFIX:?} prefix");
            };
            let Some((module, kind)) = split_pair_key(rest) else {
                bail!("unsupported V4.1 adapter tensor key {key:?}; expected .lora_A.weight/.lora_B.weight");
            };
            let target = normalize_module(module)?;
            let slot = pairs.entry(target).or_default();
            match kind {
                'A' if slot.0.replace(key.to_string()).is_some() => {
                    bail!("duplicate LoRA A tensor for module {module:?}")
                }
                'B' if slot.1.replace(key.to_string()).is_some() => {
                    bail!("duplicate LoRA B tensor for module {module:?}")
                }
                _ => {}
            }
        }
        if pairs.is_empty() {
            bail!("V4.1 LoRA adapter contains no LoRA A/B pairs");
        }
        let mut modules = BTreeMap::new();
        for (target, (a_key, b_key)) in pairs {
            let (a_key, b_key) = match (a_key, b_key) {
                (Some(a), Some(b)) => (a, b),
                _ => bail!("V4.1 LoRA module {target:?} is missing its A or B tensor"),
            };
            if !target_declared(&target, &config.target_modules) {
                bail!(
                    "V4.1 adapter targets {target:?}, which adapter_config.json does not declare"
                );
            }
            let rank = pattern_value(&target, &config.rank_pattern)
                .copied()
                .unwrap_or(config.r);
            let alpha = pattern_value(&target, &config.alpha_pattern)
                .copied()
                .unwrap_or(config.lora_alpha);
            if rank == 0 {
                bail!("V4.1 LoRA module {target:?} has rank 0");
            }
            if !alpha.is_finite() || alpha <= 0.0 {
                bail!("V4.1 LoRA module {target:?} has invalid alpha {alpha}");
            }
            let a_shape = tensors.tensor(&a_key)?.shape().to_vec();
            let b_shape = tensors.tensor(&b_key)?.shape().to_vec();
            if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[0] != rank || b_shape[1] != rank
            {
                bail!(
                    "V4.1 LoRA module {target:?} is not a rank-{rank} pair: A={a_shape:?} B={b_shape:?}"
                );
            }
            let a = decode_float_tensor(&a_key, &tensors)?;
            let b = decode_float_tensor(&b_key, &tensors)?;
            let scale = if config.use_rslora {
                alpha / (rank as f64).sqrt()
            } else {
                alpha / rank as f64
            };
            if !(scale as f32).is_finite() || scale as f32 <= 0.0 {
                bail!("V4.1 LoRA module {target:?} has a scale outside finite positive FP32");
            }
            modules.insert(
                target.clone(),
                DeepseekV41LoraModule {
                    target,
                    rank,
                    alpha,
                    scale: scale as f32,
                    a: Arc::from(a),
                    b: Arc::from(b),
                },
            );
        }
        for declared in declared_targets(&config.target_modules) {
            if !modules
                .keys()
                .any(|module| target_matches(module, &declared))
            {
                bail!(
                    "V4.1 adapter_config.json targets {declared:?}, but the adapter has no LoRA pair for it"
                );
            }
        }
        let source_sha256 = hex(&Sha256::digest(&bytes));
        Ok(Self {
            name: directory
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("default")
                .to_owned(),
            source_sha256,
            modules,
            config: config_value,
        })
    }

    /// Export an unmerged standard PEFT adapter. Frozen packed base weights are
    /// neither read nor requantized. Build the complete directory privately,
    /// then publish it with a rename; an existing destination is never replaced.
    pub fn export(&self, directory: impl AsRef<Path>) -> Result<()> {
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
        let directory = directory.as_ref();
        if directory.exists() {
            bail!(
                "V4.1 adapter export destination already exists: {}",
                directory.display()
            );
        }
        let parent = directory
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let staging = parent.join(format!(".v41-adapter-{}-{nonce}", std::process::id()));
        fs::create_dir(&staging)?;
        let result = (|| -> Result<()> {
            let mut owned = BTreeMap::new();
            for module in self.modules.values() {
                for (suffix, values, shape) in [
                    (
                        LORA_A_SUFFIX,
                        &module.a,
                        vec![module.rank, module.a.len() / module.rank],
                    ),
                    (
                        LORA_B_SUFFIX,
                        &module.b,
                        vec![module.b.len() / module.rank, module.rank],
                    ),
                ] {
                    owned.insert(
                        format!("{PEFT_STATE_PREFIX}{}{suffix}", module.target),
                        (
                            shape,
                            values
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        ),
                    );
                }
            }
            let views = owned
                .iter()
                .map(|(name, (shape, bytes))| {
                    Ok((
                        name.as_str(),
                        TensorView::new(Dtype::F32, shape.clone(), bytes)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            serialize_to_file(views, None, &staging.join(DEEPSEEK_V41_LORA_WEIGHTS))?;
            fs::write(
                staging.join(DEEPSEEK_V41_LORA_CONFIG),
                serde_json::to_vec_pretty(&self.config)?,
            )?;
            // Reuse the normal consumer's checks before making the artifact visible.
            let reloaded = Self::load(&staging)?;
            if reloaded.modules.len() != self.modules.len() {
                bail!("V4.1 exported adapter lost a module");
            }
            fs::rename(&staging, directory)
                .with_context(|| format!("publishing V4.1 adapter {}", directory.display()))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    /// Rebuild this adapter with different factor values, keeping the source
    /// config, name and checksum. The trainer publishes updated factors through
    /// this path so `export` still runs the normal consumer's validation.
    pub fn with_factors(
        &self,
        mut factors: impl FnMut(&str) -> Option<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Self> {
        let mut modules = BTreeMap::new();
        for (target, module) in &self.modules {
            let (a, b) = factors(target)
                .with_context(|| format!("V4.1 adapter factors for {target} are missing"))?;
            if a.len() != module.a.len() || b.len() != module.b.len() {
                bail!("V4.1 adapter factors for {target} changed geometry");
            }
            modules.insert(
                target.clone(),
                DeepseekV41LoraModule {
                    a: a.into(),
                    b: b.into(),
                    ..module.clone()
                },
            );
        }
        Ok(Self {
            name: self.name.clone(),
            source_sha256: self.source_sha256.clone(),
            modules,
            config: self.config.clone(),
        })
    }

    /// Bind optimizer state to the actual factor values and PEFT semantics,
    /// independent of SafeTensors header ordering or the source adapter hash.
    pub(super) fn snapshot_digest(&self) -> Result<String> {
        let mut hash = Sha256::new();
        hash.update(serde_json::to_vec(&self.config)?);
        for (target, module) in &self.modules {
            hash.update((target.len() as u64).to_le_bytes());
            hash.update(target.as_bytes());
            hash.update((module.rank as u64).to_le_bytes());
            hash.update(module.alpha.to_le_bytes());
            hash.update(module.scale.to_le_bytes());
            for values in [&module.a, &module.b] {
                hash.update((values.len() as u64).to_le_bytes());
                for value in values.iter() {
                    hash.update(value.to_le_bytes());
                }
            }
        }
        Ok(hex(&hash.finalize()))
    }

    /// Resolve the adapter module attached to one checkpoint tensor name.
    pub fn module_for_tensor(&self, tensor: &str) -> Option<&DeepseekV41LoraModule> {
        let target = tensor.strip_suffix(".weight")?;
        self.modules.get(target)
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn targets(&self) -> impl Iterator<Item = (&str, usize, f32)> + '_ {
        self.modules
            .values()
            .map(|module| (module.target.as_str(), module.rank, module.scale))
    }

    fn validate_config(config: &AdapterConfig) -> Result<()> {
        if !config.peft_type.is_empty() && !config.peft_type.eq_ignore_ascii_case("LORA") {
            bail!(
                "V4.1 native PEFT supports LORA adapters; got peft_type={:?}",
                config.peft_type
            );
        }
        if !config.task_type.is_empty() && !config.task_type.eq_ignore_ascii_case("CAUSAL_LM") {
            bail!(
                "V4.1 native PEFT requires task_type=CAUSAL_LM; got {:?}",
                config.task_type
            );
        }
        if config
            .modules_to_save
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        {
            bail!(
                "V4.1 PEFT modules_to_save has not passed its independent <=2e-7 gate; use an additive LoRA adapter without replacement tensors"
            );
        }
        if config.use_dora {
            bail!("V4.1 native PEFT does not implement DoRA (use_dora=true)");
        }
        if config.use_qalora {
            bail!("V4.1 native PEFT does not implement QALoRA (use_qalora=true)");
        }
        if config.fan_in_fan_out {
            bail!("V4.1 native PEFT does not implement fan_in_fan_out adapters");
        }
        if config.lora_bias {
            bail!("V4.1 native PEFT does not implement an independent LoRA-B bias");
        }
        if !matches!(config.bias.as_str(), "" | "none") {
            bail!(
                "V4.1 native PEFT requires additive LoRA without adapter-owned base bias; got bias={:?}",
                config.bias
            );
        }
        if config
            .target_parameters
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        {
            bail!("V4.1 native PEFT does not implement target_parameters adapters");
        }
        for (label, value) in [
            ("trainable_token_indices", &config.trainable_token_indices),
            ("alora_invocation_tokens", &config.alora_invocation_tokens),
            ("arrow_config", &config.arrow_config),
            ("layer_replication", &config.layer_replication),
        ] {
            if value.as_ref().is_some_and(|value| !value.is_null()) {
                bail!("V4.1 native PEFT does not implement {label}");
            }
        }
        if !config.lora_dropout.is_finite() || !(0.0..1.0).contains(&config.lora_dropout) {
            bail!("V4.1 LoRA dropout must be finite and in [0, 1)");
        }
        if let TargetSelector::All(value) = &config.target_modules {
            if !value.eq_ignore_ascii_case("all-linear") {
                bail!(
                    "V4.1 native PEFT requires target_modules as a module list or \"all-linear\"; regex selectors are unsupported: {value:?}"
                );
            }
        }
        if let TargetSelector::Names(names) = &config.target_modules {
            if names.is_empty() {
                bail!("V4.1 adapter_config.json must declare target_modules");
            }
            if names.iter().any(|name| name.trim().is_empty()) {
                bail!("V4.1 adapter_config.json has an empty target_modules entry");
            }
        }
        Ok(())
    }
}

/// Host-side adapter payload attached to one linear weight before upload.
/// Tensor payloads are shared with every resident or streamed copy.
#[derive(Debug, Clone)]
pub struct DeepseekV41LoraHost {
    pub rank: usize,
    pub scale: f32,
    pub a: Arc<[f32]>,
    pub b: Arc<[f32]>,
}

impl DeepseekV41LoraHost {
    pub fn from_module(module: &DeepseekV41LoraModule) -> Self {
        Self {
            rank: module.rank,
            scale: module.scale,
            a: module.a.clone(),
            b: module.b.clone(),
        }
    }
}

/// Static capability description for `architectures --peft`. Promoted only
/// alongside the recorded adapter qualification; it is never inferred from the
/// base graph's own gate.
#[derive(Debug, Clone, Copy)]
pub struct DeepseekV41PeftCapability {
    pub supports_peft: bool,
    pub validated_methods: &'static [&'static str],
    pub modules_to_save_validated: bool,
    pub exclusions: &'static str,
    pub evidence: &'static str,
}

pub const fn lora_capability() -> DeepseekV41PeftCapability {
    DeepseekV41PeftCapability {
        supports_peft: true,
        validated_methods: &["LoRA", "rsLoRA"],
        modules_to_save_validated: false,
        exclusions: "additive LoRA on packed attention/indexer/shared+routed expert projections, unmerged, zero dropout, frozen base; modules_to_save, DoRA, QALoRA, adapter bias, trainable tokens, AdamW on compressor/Engram/grouped-wo_a targets, F32-storage packages and merged-package export remain unqualified",
        evidence: "PROGRESS_DEEPSEEK_V41_AUDIT.md: production-width PEFT-LoRA logit gate at <2e-7, rtol=0, and the four-oracle full-model training gate (logits, gradients, moments, resume, export) in target/v41-completion/training-qualification.json",
    }
}

fn rejected_name(name: &str) -> Option<&'static str> {
    if name.contains("adapter_model.bin") || name.ends_with(".bin") {
        Some("pickle adapter weights")
    } else if name.ends_with(".py") {
        Some("Python adapter code")
    } else if name.ends_with(".pyc") {
        Some("compiled Python adapter code")
    } else {
        None
    }
}

fn reject_python_adapter_files(directory: &Path) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(kind) = rejected_name(&name) {
            bail!("V4.1 LoRA adapters must be plain SafeTensors; found {kind} {name:?}");
        }
    }
    Ok(())
}

/// Accept both HF module spellings: the checkpoint's `attn` and the
/// Transformers module's `self_attn`. Everything else must already match the
/// released checkpoint naming.
fn normalize_module(module: &str) -> Result<String> {
    let base_layer = module.strip_suffix(".base_layer").unwrap_or(module);
    let named = base_layer.strip_suffix(".default").unwrap_or(base_layer);
    let module = named.replace(".self_attn.", ".attn.");
    if module.is_empty()
        || module.split('.').any(|part| part.is_empty())
        || module.contains("..")
        || module.starts_with('.')
    {
        bail!("V4.1 adapter module name {module:?} is not a canonical checkpoint path");
    }
    if module.contains('/') || module.contains('\\') {
        bail!("V4.1 adapter module name {module:?} is not a canonical checkpoint path");
    }
    Ok(module)
}

fn split_pair_key(rest: &str) -> Option<(&str, char)> {
    if let Some(module) = rest.strip_suffix(".lora_A.default.weight") {
        return Some((module, 'A'));
    }
    if let Some(module) = rest.strip_suffix(".lora_B.default.weight") {
        return Some((module, 'B'));
    }
    if let Some(module) = rest.strip_suffix(LORA_A_SUFFIX) {
        return Some((module, 'A'));
    }
    if let Some(module) = rest.strip_suffix(LORA_B_SUFFIX) {
        return Some((module, 'B'));
    }
    None
}

fn declared_targets(selector: &TargetSelector) -> Vec<String> {
    match selector {
        TargetSelector::All(_) => Vec::new(),
        TargetSelector::Names(names) => names.clone(),
    }
}

fn target_declared(module: &str, selector: &TargetSelector) -> bool {
    match selector {
        TargetSelector::All(value) => value.eq_ignore_ascii_case("all-linear"),
        TargetSelector::Names(names) => names.iter().any(|name| target_matches(module, name)),
    }
}

/// PEFT matches a declared target against the module path's trailing segments.
fn target_matches(module: &str, target: &str) -> bool {
    let target = target.replace(".self_attn.", ".attn.");
    let target = target
        .strip_prefix("self_attn.")
        .map(|suffix| format!("attn.{suffix}"))
        .unwrap_or(target);
    module == target || module.ends_with(&format!(".{target}"))
}

fn pattern_value<'a, T>(module: &str, pattern: &'a BTreeMap<String, T>) -> Option<&'a T> {
    pattern
        .iter()
        .filter(|(key, _)| target_matches(module, key))
        .max_by_key(|(key, _)| key.len())
        .map(|(_, value)| value)
}

fn decode_float_tensor(
    name: &str,
    tensors: &safetensors::tensor::SafeTensors<'_>,
) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("reading adapter tensor {name:?}"))?;
    if tensor.shape().len() != 2 {
        bail!(
            "V4.1 adapter tensor {name:?} must be rank 2, got {:?}",
            tensor.shape()
        );
    }
    let data = tensor.data();
    let values: Vec<f32> = match tensor.dtype() {
        safetensors::tensor::Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        safetensors::tensor::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
            .collect(),
        safetensors::tensor::Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| half_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        other => bail!("V4.1 adapter tensor {name:?} must be F32/BF16/F16, got {other:?}"),
    };
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("V4.1 adapter tensor {name:?} must be nonempty and finite");
    }
    Ok(values)
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let value = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let shift = mantissa.leading_zeros() - 21;
            let normalized = (mantissa << shift) & 0x3ff;
            sign | ((127 - 14 - shift) << 23) | (normalized << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(value)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_module_spellings_normalize_to_the_checkpoint_path() {
        assert_eq!(
            normalize_module("layers.3.self_attn.wq_a").unwrap(),
            "layers.3.attn.wq_a"
        );
        assert_eq!(
            normalize_module("layers.3.ffn.experts.7.w1").unwrap(),
            "layers.3.ffn.experts.7.w1"
        );
        assert!(normalize_module("").is_err());
        assert!(normalize_module("layers..attn").is_err());
    }

    #[test]
    fn target_matching_follows_peft_suffix_semantics() {
        assert!(target_matches("layers.3.attn.wq_a", "wq_a"));
        assert!(target_matches("layers.3.attn.wq_a", "attn.wq_a"));
        assert!(target_matches("layers.3.attn.wq_a", "self_attn.wq_a"));
        assert!(target_matches(
            "layers.3.attn.wq_a",
            "layers.3.self_attn.wq_a"
        ));
        assert!(!target_matches("layers.3.attn.wq_a", "wq"));
        assert!(target_matches("layers.3.ffn.experts.7.w1", "w1"));
    }

    #[test]
    fn half_decoding_matches_the_ieee_conversion() {
        for (bits, expected) in [
            (0x0000u16, 0.0f32),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x3555, 0.33325195),
        ] {
            assert_eq!(half_to_f32(bits), expected);
        }
        // Include every finite half, especially subnormals in small adapters.
        for bits in 0..=u16::MAX {
            let exponent = (bits >> 10) & 31;
            if exponent == 31 {
                continue;
            }
            let mantissa = bits & 1023;
            let magnitude = if exponent == 0 {
                (mantissa as f64) * 2.0f64.powi(-24)
            } else {
                (1.0 + mantissa as f64 / 1024.0) * 2.0f64.powi(exponent as i32 - 15)
            };
            let expected = (if bits & 0x8000 == 0 {
                magnitude
            } else {
                -magnitude
            }) as f32;
            assert_eq!(
                half_to_f32(bits).to_bits(),
                expected.to_bits(),
                "half {bits:04x}"
            );
        }
    }

    #[test]
    fn unmerged_export_preserves_factors_patterns_and_rslora() -> Result<()> {
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = Cleanup(
            std::env::temp_dir().join(format!("v41-export-{}-{nonce}", std::process::id())),
        );
        for rslora in [false, true] {
            let source = root.0.join(format!("input-{rslora}"));
            let destination = root.0.join(format!("output-{rslora}"));
            fs::create_dir_all(&source)?;
            let config = serde_json::json!({
                "peft_type":"LORA", "task_type":"CAUSAL_LM", "r":2, "lora_alpha":3.5,
                "lora_dropout":0.125, "target_modules":["wq_a", "w2"], "use_rslora":rslora,
                "rank_pattern":{"ffn.shared_experts.w2":1},
                "alpha_pattern":{"ffn.shared_experts.w2":0.75},
                "base_model_name_or_path":"deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "future_metadata":{"keep":true}
            });
            fs::write(
                source.join(DEEPSEEK_V41_LORA_CONFIG),
                serde_json::to_vec(&config)?,
            )?;
            let mut owned = Vec::new();
            for (module, rank, input, output) in [
                ("layers.0.self_attn.wq_a", 2, 3, 4),
                ("layers.0.ffn.shared_experts.w2", 1, 4, 3),
            ] {
                for (suffix, shape) in [
                    ("lora_A.default.weight", vec![rank, input]),
                    ("lora_B.default.weight", vec![output, rank]),
                ] {
                    let count: usize = shape.iter().product();
                    let bytes: Vec<_> = (0..count)
                        .flat_map(|i| ((i + 1) as u16).to_le_bytes())
                        .collect();
                    owned.push((
                        format!("{PEFT_STATE_PREFIX}{module}.{suffix}"),
                        shape,
                        bytes,
                    ));
                }
            }
            let views = owned
                .iter()
                .map(|(name, shape, bytes)| {
                    Ok((
                        name.as_str(),
                        TensorView::new(Dtype::F16, shape.clone(), bytes)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            serialize_to_file(views, None, &source.join(DEEPSEEK_V41_LORA_WEIGHTS))?;
            let adapter = DeepseekV41LoraAdapter::load(&source)?;
            adapter.export(&destination)?;
            let reloaded = DeepseekV41LoraAdapter::load(&destination)?;
            assert_eq!(reloaded.config, config);
            assert_eq!(reloaded.len(), adapter.len());
            for (target, module) in &adapter.modules {
                let copy = &reloaded.modules[target];
                assert_eq!(copy.rank, module.rank);
                assert_eq!(copy.alpha, module.alpha);
                assert_eq!(copy.scale.to_bits(), module.scale.to_bits());
                assert!(copy
                    .a
                    .iter()
                    .zip(&*module.a)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(copy
                    .b
                    .iter()
                    .zip(&*module.b)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            let before = fs::read(destination.join(DEEPSEEK_V41_LORA_WEIGHTS))?;
            assert!(adapter.export(&destination).is_err());
            assert_eq!(
                fs::read(destination.join(DEEPSEEK_V41_LORA_WEIGHTS))?,
                before
            );
        }
        Ok(())
    }

    #[test]
    fn capability_stays_fail_closed_for_replacement_modules() {
        let capability = lora_capability();
        assert!(capability.supports_peft);
        assert!(!capability.modules_to_save_validated);
        assert_eq!(capability.validated_methods, &["LoRA", "rsLoRA"]);
    }
}
