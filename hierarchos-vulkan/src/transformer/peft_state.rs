use super::*;
use sha2::{Digest, Sha256};

const OPTIMIZER_FILE: &str = "hierarchos_adapter_optimizer.safetensors";

/// Adapter-local replacement state for PEFT `modules_to_save` linears.
///
/// The LoRA bank stays attached to the live `VulkanLinear`; only the wrapped
/// base parameter identity and its trainability flags move between the
/// canonical base and named-adapter banks.
#[derive(Clone)]
pub(super) struct PeftSavedLinearState {
    weight: TrainableParameter,
    bias: TrainableParameter,
    train_base: bool,
    train_bias: bool,
}

impl PeftSavedLinearState {
    fn adapter(linear: &VulkanLinear) -> Self {
        Self {
            weight: linear.weight.clone(),
            bias: linear.bias.clone(),
            train_base: linear.train_base,
            train_bias: linear.train_bias,
        }
    }

    fn frozen_base(linear: &VulkanLinear) -> Self {
        Self {
            weight: linear.weight.frozen_peft_base_handle(),
            bias: linear.bias.frozen_peft_base_handle(),
            train_base: false,
            train_bias: false,
        }
    }

    fn install(&self, linear: &mut VulkanLinear) {
        linear.weight = self.weight.clone();
        linear.bias = self.bias.clone();
        linear.train_base = self.train_base;
        linear.train_bias = self.train_bias;
    }
}

#[derive(Clone)]
pub(super) struct PeftSavedNormState {
    weight: TrainableParameter,
    bias: TrainableParameter,
    trainable: bool,
    train_bias: bool,
}

impl PeftSavedNormState {
    fn adapter(norm: &VulkanLayerNorm) -> Self {
        Self {
            weight: norm.weight.clone(),
            bias: norm.bias.clone(),
            trainable: norm.trainable,
            train_bias: norm.train_bias,
        }
    }

    fn frozen_base(norm: &VulkanLayerNorm) -> Self {
        Self {
            weight: norm.weight.frozen_peft_base_handle(),
            bias: norm.bias.frozen_peft_base_handle(),
            trainable: false,
            train_bias: false,
        }
    }

    fn install(&self, norm: &mut VulkanLayerNorm) {
        norm.weight = self.weight.clone();
        norm.bias = self.bias.clone();
        norm.trainable = self.trainable;
        norm.train_bias = self.train_bias;
    }
}

#[derive(Clone)]
pub(super) enum PeftSavedLmHeadState {
    /// The canonical graph used the tied `shared_embedding` output path and
    /// therefore had no independent `VulkanLinear` lm_head object.
    Tied,
    Linear(PeftSavedLinearState),
}

#[derive(Clone, Default)]
pub(super) struct PeftSavedModuleState {
    linears: BTreeMap<String, PeftSavedLinearState>,
    norms: BTreeMap<String, PeftSavedNormState>,
    lm_head: Option<PeftSavedLmHeadState>,
    embedding: Option<(String, SharedLmHeadParameter)>,
}

impl PeftSavedModuleState {
    fn is_empty(&self) -> bool {
        self.linears.is_empty()
            && self.norms.is_empty()
            && self.lm_head.is_none()
            && self.embedding.is_none()
    }
}

#[derive(Serialize, Deserialize)]
struct LoraOptimizerMetadata {
    version: u32,
    step: u32,
    #[serde(default)]
    embedding_step: Option<u32>,
    dropout_seed: u32,
    batch_size: usize,
    seq_len: usize,
    base_sha256: String,
    adapter_sha256: String,
    adapter_config: serde_json::Value,
    hyper: [f32; 5],
}

fn validate_optimizer_hyper(hyper: [f32; 5]) -> Result<()> {
    let [lr, beta1, beta2, eps, decay] = hyper;
    if hyper.iter().any(|v| !v.is_finite()) || lr < 0.0 || eps <= 0.0 || decay < 0.0
        || !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
        bail!("invalid LoRA checkpoint AdamW settings");
    }
    Ok(())
}

impl VulkanTransformer {
    /// Number of native trainable elements, without downloading GPU tensors.
    pub fn lora_trainable_parameter_count(&self) -> Result<usize> {
        let mut count = self.lora_optimizer_parameters()?.values().map(|p| p.len).sum();
        if self.lora_saved_embedding_module.is_some() {
            count += self.shared_embedding.context_dim() * self.shared_embedding.vocab_size();
        }
        Ok(count)
    }

    /// Plan a uniform standard-LoRA rank from the actual native module graph.
    /// The denominator is serialized floating base tensor elements, with a
    /// duplicate tied output head counted only once. No weight data is read.
    pub fn lora_rank_for_parameter_budget(
        &mut self, config: &VulkanTransformerLoraConfig, model_dir: impl AsRef<Path>,
        percent: f64,
    ) -> Result<(usize, usize, usize)> {
        if !percent.is_finite() || percent <= 0.0 || percent > 100.0 {
            bail!("PEFT parameter percent must be finite and in (0, 100]");
        }
        if config.bias != "none" || config.use_dora || config.lora_bias
            || !config.rank_pattern.is_empty()
            || config.modules_to_save.as_ref().is_some_and(|v| !v.is_empty())
            || config.target_parameters.as_ref().is_some_and(|v| !v.is_empty())
            || config.trainable_token_indices.is_some() || config.layer_replication.is_some() {
            bail!("automatic rank budgeting supports uniform LoRA without bias, DoRA, rank_pattern, modules_to_save, trainable tokens, packed parameters or layer replication; use an explicit rank for those configurations");
        }
        self.count_lora_target_modules(config)?;
        let store = Gpt2TensorStore::load(model_dir.as_ref())?;
        let base_count = store.tensors.iter()
            .filter(|(name, _)| !(self.config.tie_word_embeddings && name.ends_with("lm_head.weight")))
            .try_fold(0usize, |total, (_, tensor)| {
                let size = tensor.shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d))
                    .context("base tensor shape overflow")?;
                total.checked_add(size).context("base parameter count overflow")
            })?;
        let mut per_rank = 0usize;
        self.for_each_verified_peft_linear_mut(|_, name, linear| {
            if config.targets(&name) {
                per_rank = per_rank.checked_add(linear.input_dim + linear.output_dim)
                    .context("LoRA parameter count overflow")?;
            }
            Ok(())
        })?;
        if per_rank == 0 { bail!("parameter budget selected no native LoRA linears"); }
        let budget = (base_count as f64 * percent / 100.0).floor() as usize;
        let rank = budget / per_rank;
        if rank == 0 {
            bail!("parameter budget {percent}% allows {budget} elements, but rank 1 requires {per_rank}; increase the budget or narrow target modules");
        }
        Ok((rank, base_count, budget))
    }

    pub(super) fn save_peft_progress(&mut self) {
        if self.lora_adapters_enabled {
            if let Some(name) = self.lora_active_adapter_name.as_ref() {
                self.lora_progress.insert(name.clone(), (self.step, self.dropout_seed));
            }
        }
    }

    /// Canonical native parameters for ordinary LoRA checkpointing. This uses
    /// the same module traversal as PEFT export; no architecture-specific
    /// optimizer is introduced. Replacement wrappers need additional state and
    /// must not silently produce an incomplete training checkpoint.
    fn lora_optimizer_parameters(&self) -> Result<BTreeMap<String, &TrainableParameter>> {
        let config = self.lora_config.as_ref().context("no active LoRA adapter")?;
        if !self.lora_adapters_enabled || config.inference_mode {
            bail!("optimizer checkpoint requires an enabled, trainable LoRA adapter");
        }
        if !config.bias.eq_ignore_ascii_case("none")
            || config.trainable_token_indices.is_some()
            || config.target_parameters.as_ref().is_some_and(|v| !v.is_empty())
            || config.layer_replication.is_some()
        {
            bail!("optimizer checkpoint for bias, tokens, packed parameters or replicated layers is not implemented");
        }
        if config.modules_to_save.as_ref().is_some_and(|v| !v.is_empty())
            && !self.supports_verified_modules_to_save_bank() {
            bail!("optimizer checkpoint requires a qualified modules_to_save bank");
        }
        let mut parameters = BTreeMap::new();
        for index in 0..self.layers.len() {
            for (name, linear) in self.peft_layer_linears(index)? {
                if let Some(lora) = linear.lora.as_ref() {
                    if !lora.trainable { bail!("adapter module {name} is not trainable"); }
                    for (suffix, parameter) in [("lora_A", &lora.a), ("lora_B", &lora.b)] {
                        parameters.insert(format!("{name}.{suffix}.weight"), parameter);
                    }
                    if let Some(bias) = &lora.b_bias {
                        parameters.insert(format!("{name}.lora_B.bias"), bias);
                    }
                    if let Some(dora) = &lora.dora {
                        parameters.insert(format!("{name}.lora_magnitude_vector"), &dora.magnitude);
                    }
                }
            }
            if matches!(
                self.config.architecture,
                VulkanTransformerArchitecture::Qwen3Next
                    | VulkanTransformerArchitecture::Qwen35
                    | VulkanTransformerArchitecture::Qwen35Moe
                    | VulkanTransformerArchitecture::Qwen4Exp
            ) && self.layers[index].qwen_gated_delta.is_none()
            {
                let name = format!("model.layers.{index}.self_attn.q_proj");
                if config.targets(&name) {
                    let query = self.layers[index]
                        .q_proj
                        .as_ref()
                        .context("Qwen hybrid optimizer state is missing q_proj")?;
                    let gate = self.layers[index].attention_gate.as_ref().context(
                        "Qwen hybrid optimizer state is missing packed q_proj gate",
                    )?;
                    let query_lora = query
                        .lora
                        .as_ref()
                        .context("Qwen hybrid optimizer state is missing query LoRA")?;
                    let gate_lora = gate
                        .lora
                        .as_ref()
                        .context("Qwen hybrid optimizer state is missing gate LoRA")?;
                    if !query_lora.trainable || !gate_lora.trainable {
                        bail!("adapter module {name} is not trainable");
                    }
                    if query_lora.a.read()? != gate_lora.a.read()? {
                        bail!("Qwen hybrid q_proj split LoRA-A state diverged");
                    }
                    parameters.insert(format!("{name}.lora_A.weight"), &gate_lora.a);
                    parameters.insert(format!("{name}.lora_B.query.weight"), &query_lora.b);
                    parameters.insert(format!("{name}.lora_B.gate.weight"), &gate_lora.b);
                    match (query_lora.b_bias.as_ref(), gate_lora.b_bias.as_ref()) {
                        (Some(query_bias), Some(gate_bias)) => {
                            parameters.insert(
                                format!("{name}.lora_B.query.bias"),
                                query_bias,
                            );
                            parameters.insert(format!("{name}.lora_B.gate.bias"), gate_bias);
                        }
                        (None, None) => {}
                        _ => bail!("Qwen hybrid q_proj split LoRA-B bias state mismatch"),
                    }
                    match (query_lora.dora.as_ref(), gate_lora.dora.as_ref()) {
                        (Some(query_dora), Some(gate_dora)) => {
                            parameters.insert(
                                format!("{name}.lora_magnitude_vector.query"),
                                &query_dora.magnitude,
                            );
                            parameters.insert(
                                format!("{name}.lora_magnitude_vector.gate"),
                                &gate_dora.magnitude,
                            );
                        }
                        (None, None) => {}
                        _ => bail!("Qwen hybrid q_proj split DoRA state mismatch"),
                    }
                }
            }
        }
        if let Some(name) = self.lora_active_adapter_name.as_ref() {
            if let Some(saved) = self.lora_saved_module_adapters.get(name) {
                for (module, linear) in &saved.linears {
                    parameters.insert(format!("{module}.saved.weight"), &linear.weight);
                    if linear.train_bias { parameters.insert(format!("{module}.saved.bias"), &linear.bias); }
                }
                for (module, norm) in &saved.norms {
                    parameters.insert(format!("{module}.saved.weight"), &norm.weight);
                    if norm.train_bias { parameters.insert(format!("{module}.saved.bias"), &norm.bias); }
                }
                if let Some(PeftSavedLmHeadState::Linear(head)) = &saved.lm_head {
                    parameters.insert("lm_head.saved.weight".into(), &head.weight);
                    if head.train_bias { parameters.insert("lm_head.saved.bias".into(), &head.bias); }
                }
            }
        }
        if parameters.is_empty() { bail!("adapter optimizer has no parameters"); }
        Ok(parameters)
    }

    pub(super) fn lora_optimizer_buffers(&self) -> Result<BTreeMap<String, (usize, GpuBuffer, GpuBuffer)>> {
        let mut buffers = BTreeMap::new();
        for (name, parameter) in self.lora_optimizer_parameters()? {
            buffers.insert(name, (parameter.len,
                parameter.exp_avg.as_ref().context("missing first moment")?.clone(),
                parameter.exp_avg_sq.as_ref().context("missing second moment")?.clone()));
        }
        if let Some(module) = &self.lora_saved_embedding_module {
            let (avg, sq) = self.shared_embedding.peft_optimizer_buffers()?;
            buffers.insert(format!("{module}.saved.weight"), (self.shared_embedding.context_dim() * self.shared_embedding.vocab_size(), avg, sq));
        }
        Ok(buffers)
    }

    /// Last native adapter gradients in the same canonical layout as HF export.
    /// Reads Vulkan buffers; this does not run or approximate backward on CPU.
    pub fn lora_gradient_values(&self) -> Result<BTreeMap<String, Vec<f32>>> {
        let mut result = BTreeMap::new();
        let config = self.lora_config.as_ref().context("missing adapter config")?;
        let mut saved_shapes = BTreeMap::new();
        for index in 0..self.layers.len() {
            for (name, linear) in self.peft_layer_linears(index)? {
                saved_shapes.insert(name, (linear.output_dim, linear.input_dim));
            }
        }
        let parameters = self.lora_optimizer_parameters()?;
        let mut consumed = BTreeSet::new();
        for (name, parameter) in &parameters {
            if consumed.contains(name) {
                continue;
            }
            if let Some(module) = name.strip_suffix(".lora_B.query.weight") {
                let gate_name = format!("{module}.lora_B.gate.weight");
                let gate = parameters.get(&gate_name).with_context(|| {
                    format!("Qwen hybrid gradient state is missing {gate_name}")
                })?;
                let query_values = parameter.grad_buffer()?.read_f32(parameter.len)?;
                let gate_values = gate.grad_buffer()?.read_f32(gate.len)?;
                let packed = pack_qwen_hybrid_query_gate(
                    &query_values,
                    &gate_values,
                    config.r,
                    self.config.num_heads,
                    self.config.head_dim,
                )?;
                result.insert(
                    format!("base_model.model.{module}.lora_B.weight"),
                    packed,
                );
                consumed.insert(name.clone());
                consumed.insert(gate_name);
                continue;
            }
            if name.ends_with(".lora_B.gate.weight") {
                continue;
            }
            let mut values = parameter.grad_buffer()?.read_f32(parameter.len)?;
            if config.fan_in_fan_out {
                if let Some(module) = name.strip_suffix(".saved.weight") {
                    if let Some(&(rows, cols)) = saved_shapes.get(module) {
                        values = transpose(&values, rows, cols);
                    }
                }
            }
            result.insert(format!("base_model.model.{}", name.replace(".saved.", ".")), values);
        }
        if let Some(module) = &self.lora_saved_embedding_module {
            let len = self.shared_embedding.context_dim() * self.shared_embedding.vocab_size();
            result.insert(format!("base_model.model.{module}.weight"), self.shared_embedding.gradient_buffer()?.read_f32(len)?);
        }
        Ok(result)
    }

    fn lora_base_fingerprint(&mut self) -> Result<String> {
        let selected = self.lora_active_adapter_name.clone();
        let restore = self.lora_adapters_enabled && !self.lora_saved_module_base.is_empty();
        if restore { self.restore_peft_saved_module_base()?; }
        let result = (|| -> Result<String> {
            let mut digest = Sha256::new();
            digest.update(serde_json::to_vec(&self.config)?);
            // PLE tables are file-backed and can be tens of GiB. Hash their
            // source ranges in bounded chunks rather than materializing them
            // through parameter_values (which deliberately rejects PLE).
            let resident = if self.config.architecture == VulkanTransformerArchitecture::Qwen4Exp {
                self.qwen4_parameter_tensors()?.into_iter()
                    .map(|(name, (_, values))| (name, values)).collect()
            } else {
                self.parameter_values()?
            };
            for (name, values) in resident {
                digest.update((name.len() as u64).to_le_bytes());
                digest.update(name.as_bytes());
                digest.update((values.len() as u64).to_le_bytes());
                digest.update(f32_le_bytes(&values));
            }
            for layer in &self.layers {
                if let Some(ple) = &layer.qwen4_ple {
                    let source = &ple.embedding_source;
                    digest.update(serde_json::to_vec(&(
                        &ple.embedding_name, &source.shape, format!("{:?}", source.dtype), source.byte_len,
                    ))?);
                    let mut file = fs::File::open(&source.path)?;
                    file.seek(SeekFrom::Start(source.data_start))?;
                    let mut remaining = source.byte_len;
                    let mut chunk = [0u8; 65536];
                    while remaining != 0 {
                        let len = remaining.min(chunk.len());
                        file.read_exact(&mut chunk[..len])?;
                        digest.update(&chunk[..len]);
                        remaining -= len;
                    }
                    let mut state = ple.sparse_embedding_state.lock()
                        .map_err(|_| anyhow!("Qwen4 PLE sparse embedding state mutex is poisoned"))?;
                    // Inference populates a row cache without changing weights.
                    // Only actual optimizer updates belong to base identity.
                    if !state.history.is_empty() {
                        let (decay, rows) = state.export_snapshot()?;
                        digest.update(decay.to_le_bytes());
                        for (index, values) in rows {
                            digest.update((index as u64).to_le_bytes());
                            digest.update(f32_le_bytes(&values));
                        }
                    }
                }
            }
            Ok(format!("{:x}", digest.finalize()))
        })();
        if restore { self.activate_peft_saved_module_adapter(selected.as_deref().context("missing selected adapter")?)?; }
        result
    }

    /// Save the standard HF adapter plus native AdamW/RNG state into a new
    /// directory. The caller owns the learning-rate schedule and data cursor.
    /// No frozen base tensors are written to the checkpoint.
    pub fn save_lora_training_checkpoint(
        &mut self, output: impl AsRef<Path>, hyper: AdamWHyperParams,
    ) -> Result<()> {
        let output = output.as_ref();
        validate_optimizer_hyper([hyper.lr, hyper.beta1, hyper.beta2, hyper.eps, hyper.weight_decay])?;
        if output.exists() { bail!("training checkpoint destination already exists: {}", output.display()); }
        let parameters = self.lora_optimizer_buffers()?;
        let mut owned = BTreeMap::<String, Vec<u8>>::new();
        for (name, (len, avg, sq)) in parameters {
            owned.insert(format!("{name}.exp_avg"), f32_le_bytes(&avg.read_f32(len)?));
            owned.insert(format!("{name}.exp_avg_sq"), f32_le_bytes(&sq.read_f32(len)?));
        }
        // Prepare and validate all device reads before creating the destination.
        let base_sha256 = self.lora_base_fingerprint()?;
        let adapter_config = serde_json::to_value(self.lora_config.as_ref().context("missing adapter config")?)?;
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .with_context(|| format!("creating PEFT checkpoint parent {}", parent.display()))?;
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let staging = parent.join(format!(".peft-checkpoint-{}-{nonce}", std::process::id()));
        fs::create_dir(&staging)
            .with_context(|| format!("creating PEFT checkpoint staging directory {}", staging.display()))?;
        let result = (|| -> Result<()> {
            self.export_lora_adapter(&staging)
                .with_context(|| format!("exporting PEFT adapter into {}", staging.display()))?;
            let adapter_weights = staging.join(VULKAN_TRANSFORMER_ADAPTER_WEIGHTS_FILENAME);
            let adapter_sha256 = format!(
                "{:x}",
                Sha256::digest(
                    fs::read(&adapter_weights)
                        .with_context(|| format!("reading staged PEFT adapter {}", adapter_weights.display()))?,
                ),
            );
            let state = LoraOptimizerMetadata {
                version: 1, step: self.step, embedding_step: self.lora_saved_embedding_module.as_ref().map(|_| self.shared_embedding.step()), dropout_seed: self.dropout_seed,
                batch_size: self.batch_size, seq_len: self.seq_len,
                base_sha256, adapter_sha256, adapter_config,
                hyper: [hyper.lr, hyper.beta1, hyper.beta2, hyper.eps, hyper.weight_decay],
            };
            let views = owned.iter().map(|(name, bytes)| {
                Ok((name.as_str(), TensorView::new(Dtype::F32, vec![bytes.len()/4], bytes)?))
            }).collect::<Result<Vec<_>>>()?;
            let optimizer_path = staging.join(OPTIMIZER_FILE);
            serialize_to_file(views, Some(std::collections::HashMap::from([
                ("hierarchos_lora_optimizer".to_owned(), serde_json::to_string(&state)?)
            ])), &optimizer_path)
                .with_context(|| format!("serializing PEFT optimizer state {}", optimizer_path.display()))?;
            // Windows can transiently deny the directory rename immediately
            // after safetensors closes its newly-created files (for example
            // while Defender/indexing has a short-lived handle).  The
            // destination is required to be absent above, so retry only this
            // final atomic publish and only for PermissionDenied; every other
            // filesystem error remains fail-closed.
            let mut publish_error = None;
            for attempt in 0..8 {
                match fs::rename(&staging, output) {
                    Ok(()) => {
                        publish_error = None;
                        break;
                    }
                    Err(error)
                        if cfg!(windows)
                            && error.kind() == std::io::ErrorKind::PermissionDenied
                            && attempt < 7 =>
                    {
                        publish_error = Some(error);
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => {
                        publish_error = Some(error);
                        break;
                    }
                }
            }
            if let Some(error) = publish_error {
                return Err(error).with_context(|| {
                    format!(
                        "publishing PEFT checkpoint {} -> {}",
                        staging.display(),
                        output.display(),
                    )
                });
            }
            Ok(())
        })();
        if result.is_err() { fs::remove_dir_all(&staging).context("cleaning incomplete PEFT checkpoint")?; }
        result
    }

    /// Restore AdamW moments and RNG progress after loading this checkpoint's
    /// adapter for training. Validates the full payload and parameter identity
    /// before writing any device state. Returns the saved optimizer settings.
    pub fn load_lora_optimizer_state(&mut self, checkpoint: impl AsRef<Path>) -> Result<AdamWHyperParams> {
        let checkpoint = checkpoint.as_ref();
        let bytes = fs::read(checkpoint.join(OPTIMIZER_FILE))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let (_, header) = SafeTensors::read_metadata(&bytes)?;
        let metadata = header.metadata().as_ref().and_then(|m| m.get("hierarchos_lora_optimizer"))
            .context("missing LoRA optimizer metadata")?;
        let state: LoraOptimizerMetadata = serde_json::from_str(metadata)?;
        validate_optimizer_hyper(state.hyper)?;
        if state.version != 1 || state.batch_size != self.batch_size || state.seq_len != self.seq_len {
            bail!("LoRA optimizer checkpoint version or training geometry mismatch");
        }
        let current_base_sha256 = self.lora_base_fingerprint()?;
        if state.base_sha256 != current_base_sha256 {
            bail!(
                "LoRA optimizer checkpoint base model identity mismatch: saved={} current={}",
                state.base_sha256,
                current_base_sha256
            );
        }
        if state.adapter_config != serde_json::to_value(self.lora_config.as_ref().context("missing adapter config")?)? {
            bail!("LoRA optimizer checkpoint adapter configuration mismatch");
        }
        let weights = fs::read(checkpoint.join(VULKAN_TRANSFORMER_ADAPTER_WEIGHTS_FILENAME))?;
        if state.adapter_sha256 != format!("{:x}", Sha256::digest(&weights)) {
            bail!("LoRA optimizer checkpoint adapter weights mismatch");
        }
        let parameters = self.lora_optimizer_buffers()?;
        let mut pending = Vec::new();
        let mut expected = BTreeSet::new();
        if state.embedding_step.is_some() != self.lora_saved_embedding_module.is_some() {
            bail!("LoRA optimizer checkpoint embedding state mismatch");
        }
        for (name, (len, avg, sq)) in parameters {
            for (suffix, buffer) in [("exp_avg", avg), ("exp_avg_sq", sq)] {
                let key = format!("{name}.{suffix}");
                let tensor = tensors.tensor(&key)?;
                if tensor.dtype() != Dtype::F32 || tensor.shape() != [len] {
                    bail!("LoRA optimizer state geometry mismatch for {key}");
                }
                let values = tensor.data().chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
                if values.iter().any(|v| !v.is_finite() || (suffix == "exp_avg_sq" && *v < 0.0)) {
                    bail!("invalid LoRA optimizer moments for {key}");
                }
                pending.push((buffer, values));
                expected.insert(key);
            }
        }
        if tensors.names().into_iter().map(str::to_owned).collect::<BTreeSet<_>>() != expected {
            bail!("LoRA optimizer checkpoint tensor set mismatch");
        }
        // Validate the loaded adapter itself using export's canonical layout,
        // including architecture-specific fused projection permutations.
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let scratch = std::env::temp_dir().join(format!("hierarchos-peft-verify-{}-{nonce}", std::process::id()));
        fs::create_dir(&scratch)?;
        let identity = (|| -> Result<bool> {
            self.export_lora_adapter(&scratch)?;
            Ok(state.adapter_sha256 == format!("{:x}", Sha256::digest(fs::read(scratch.join(VULKAN_TRANSFORMER_ADAPTER_WEIGHTS_FILENAME))?)))
        })();
        fs::remove_dir_all(&scratch)?;
        if !identity? { bail!("loaded LoRA parameters do not match the optimizer checkpoint"); }
        for (buffer, values) in pending { buffer.write_f32(&values)?; }
        if let Some(step) = state.embedding_step { self.shared_embedding.restore_peft_optimizer_step(step); }
        self.step = state.step;
        self.dropout_seed = state.dropout_seed;
        let [lr, beta1, beta2, eps, weight_decay] = state.hyper;
        Ok(AdamWHyperParams { lr, beta1, beta2, eps, weight_decay })
    }
}

impl VulkanTransformer {
    /// The strict PEFT audit currently has independent <=2e-7 base and PEFT
    /// oracle evidence for exactly these families. Keep multi-adapter
    /// `modules_to_save` on that verified surface until another architecture
    /// is promoted by the same audit path.
    pub(super) fn supports_verified_modules_to_save_bank(&self) -> bool {
        self.config.architecture.peft_capability().replacement_bank_available
    }

    /// Visit every ordinary native linear that participates in the verified
    /// PEFT topology. `storage_key` is unique even when one HF module maps to
    /// multiple native linears (Qwen3-Next packed q_proj); `module` remains the
    /// canonical HF name used for PEFT suffix matching.
    fn for_each_verified_peft_linear_mut<F>(&mut self, mut visit: F) -> Result<()>
    where
        F: FnMut(String, String, &mut VulkanLinear) -> Result<()>,
    {
        match self.config.architecture {
            VulkanTransformerArchitecture::Gpt2 => {
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    let c_attn_name = format!("transformer.h.{index}.attn.c_attn");
                    let c_attn = layer
                        .c_attn
                        .as_mut()
                        .context("GPT-2 layer is missing c_attn")?;
                    visit(c_attn_name.clone(), c_attn_name, c_attn)?;

                    let name = format!("transformer.h.{index}.attn.c_proj");
                    visit(name.clone(), name, &mut layer.c_proj)?;
                    let name = format!("transformer.h.{index}.mlp.c_fc");
                    visit(name.clone(), name, &mut layer.c_fc)?;
                    let name = format!("transformer.h.{index}.mlp.c_proj");
                    visit(name.clone(), name, &mut layer.c_mlp_proj)?;
                }
            }
            VulkanTransformerArchitecture::Llama => {
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    let p = format!("model.layers.{index}");
                    let name = format!("{p}.self_attn.q_proj");
                    visit(
                        name.clone(),
                        name,
                        layer.q_proj.as_mut().context("Llama layer is missing q_proj")?,
                    )?;
                    let name = format!("{p}.self_attn.k_proj");
                    visit(
                        name.clone(),
                        name,
                        layer.k_proj.as_mut().context("Llama layer is missing k_proj")?,
                    )?;
                    let name = format!("{p}.self_attn.v_proj");
                    visit(
                        name.clone(),
                        name,
                        layer.v_proj.as_mut().context("Llama layer is missing v_proj")?,
                    )?;
                    let name = format!("{p}.self_attn.o_proj");
                    visit(name.clone(), name, &mut layer.c_proj)?;
                    if let Some(gate) = layer.c_gate.as_mut() {
                        let name = format!("{p}.mlp.gate_proj");
                        visit(name.clone(), name, gate)?;
                    }
                    let name = format!("{p}.mlp.up_proj");
                    visit(name.clone(), name, &mut layer.c_fc)?;
                    let name = format!("{p}.mlp.down_proj");
                    visit(name.clone(), name, &mut layer.c_mlp_proj)?;
                }
            }
            VulkanTransformerArchitecture::Mixtral => {
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    let p = format!("model.layers.{index}");
                    let name = format!("{p}.self_attn.q_proj");
                    visit(
                        name.clone(),
                        name,
                        layer.q_proj.as_mut().context("Mixtral layer is missing q_proj")?,
                    )?;
                    let name = format!("{p}.self_attn.k_proj");
                    visit(
                        name.clone(),
                        name,
                        layer.k_proj.as_mut().context("Mixtral layer is missing k_proj")?,
                    )?;
                    let name = format!("{p}.self_attn.v_proj");
                    visit(
                        name.clone(),
                        name,
                        layer.v_proj.as_mut().context("Mixtral layer is missing v_proj")?,
                    )?;
                    let name = format!("{p}.self_attn.o_proj");
                    visit(name.clone(), name, &mut layer.c_proj)?;

                    let sparse = format!("{p}.block_sparse_moe");
                    {
                        let moe = layer
                            .moe
                            .as_mut()
                            .context("Mixtral layer is missing MoE parameters")?;
                        let name = format!("{sparse}.gate");
                        visit(name.clone(), name, &mut moe.router)?;
                    }
                    let name = format!("{sparse}.experts.0.w1");
                    visit(
                        name.clone(),
                        name,
                        layer.c_gate.as_mut().context("Mixtral expert 0 is missing w1")?,
                    )?;
                    let name = format!("{sparse}.experts.0.w2");
                    visit(name.clone(), name, &mut layer.c_mlp_proj)?;
                    let name = format!("{sparse}.experts.0.w3");
                    visit(name.clone(), name, &mut layer.c_fc)?;
                    let moe = layer
                        .moe
                        .as_mut()
                        .context("Mixtral layer is missing MoE parameters")?;
                    for (expert_offset, expert) in moe.experts.iter_mut().enumerate() {
                        let expert_index = expert_offset + 1;
                        let expert_prefix = format!("{sparse}.experts.{expert_index}");
                        let name = format!("{expert_prefix}.w1");
                        visit(
                            name.clone(),
                            name,
                            expert.gate.as_mut().context("Mixtral expert is missing w1")?,
                        )?;
                        let name = format!("{expert_prefix}.w2");
                        visit(name.clone(), name, &mut expert.down)?;
                        let name = format!("{expert_prefix}.w3");
                        visit(name.clone(), name, &mut expert.up)?;
                    }
                }
            }
            VulkanTransformerArchitecture::Qwen3Next => {
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    let p = format!("model.layers.{index}");
                    if let Some(gated_delta) = layer.qwen_gated_delta.as_mut() {
                        match &mut gated_delta.projection {
                            VulkanQwenGatedDeltaProjection::Qwen3Next { qkvz, ba, .. } => {
                                let name = format!("{p}.linear_attn.in_proj_qkvz");
                                visit(name.clone(), name, qkvz)?;
                                let name = format!("{p}.linear_attn.in_proj_ba");
                                visit(name.clone(), name, ba)?;
                            }
                            _ => bail!(
                                "verified Qwen3-Next PEFT graph contains a non-Qwen3-Next KDA projection"
                            ),
                        }
                        let name = format!("{p}.linear_attn.out_proj");
                        visit(name.clone(), name, &mut layer.c_proj)?;
                    } else {
                        let q_name = format!("{p}.self_attn.q_proj");
                        visit(
                            format!("{q_name}::query"),
                            q_name.clone(),
                            layer
                                .q_proj
                                .as_mut()
                                .context("Qwen3-Next full-attention layer is missing q_proj")?,
                        )?;
                        visit(
                            format!("{q_name}::gate"),
                            q_name,
                            layer.attention_gate.as_mut().context(
                                "Qwen3-Next full-attention layer is missing packed q_proj gate",
                            )?,
                        )?;
                        let name = format!("{p}.self_attn.k_proj");
                        visit(
                            name.clone(),
                            name,
                            layer
                                .k_proj
                                .as_mut()
                                .context("Qwen3-Next full-attention layer is missing k_proj")?,
                        )?;
                        let name = format!("{p}.self_attn.v_proj");
                        visit(
                            name.clone(),
                            name,
                            layer
                                .v_proj
                                .as_mut()
                                .context("Qwen3-Next full-attention layer is missing v_proj")?,
                        )?;
                        let name = format!("{p}.self_attn.o_proj");
                        visit(name.clone(), name, &mut layer.c_proj)?;
                    }

                    if let Some(moe) = layer.moe.as_mut() {
                        if let Some(shared) = moe.shared_expert.as_mut() {
                            let name = format!("{p}.mlp.shared_expert.gate_proj");
                            visit(
                                name.clone(),
                                name,
                                shared.gate.as_mut().context(
                                    "Qwen3-Next shared expert is missing gate_proj",
                                )?,
                            )?;
                            let name = format!("{p}.mlp.shared_expert.up_proj");
                            visit(name.clone(), name, &mut shared.up)?;
                            let name = format!("{p}.mlp.shared_expert.down_proj");
                            visit(name.clone(), name, &mut shared.down)?;
                        }
                        if let Some(shared_gate) = moe.shared_expert_gate.as_mut() {
                            let name = format!("{p}.mlp.shared_expert_gate");
                            visit(name.clone(), name, shared_gate)?;
                        }
                    } else {
                        let name = format!("{p}.mlp.gate_proj");
                        visit(
                            name.clone(),
                            name,
                            layer
                                .c_gate
                                .as_mut()
                                .context("Qwen3-Next dense MLP is missing gate_proj")?,
                        )?;
                        let name = format!("{p}.mlp.up_proj");
                        visit(name.clone(), name, &mut layer.c_fc)?;
                        let name = format!("{p}.mlp.down_proj");
                        visit(name.clone(), name, &mut layer.c_mlp_proj)?;
                    }
                }
            }
            _ => self.for_each_graph_peft_linear_mut(visit)?,
        }
        Ok(())
    }

    /// Reuse canonical graph discovery for replacement storage. Addresses are
    /// identity keys only: no raw pointer is dereferenced, and no graph node is
    /// moved while the mutable visitor runs. Separate packed projection pieces
    /// retain separate storage keys while sharing the HF module name.
    fn for_each_graph_peft_linear_mut<F>(&mut self, mut visit: F) -> Result<()>
    where F: FnMut(String, String, &mut VulkanLinear) -> Result<()> {
        let mut names = BTreeMap::new();
        for index in 0..self.layers.len() {
            for (name, linear) in self.peft_layer_linears(index)? {
                names.insert(std::ptr::from_ref(linear) as usize, (name.clone(), name));
            }
            if matches!(self.config.architecture, VulkanTransformerArchitecture::Qwen35
                | VulkanTransformerArchitecture::Qwen35Moe | VulkanTransformerArchitecture::Qwen4Exp)
                && self.layers[index].qwen_gated_delta.is_none() {
                let name = format!("model.layers.{index}.self_attn.q_proj");
                for (suffix, linear) in [("query", &self.layers[index].q_proj),
                                        ("gate", &self.layers[index].attention_gate)] {
                    if let Some(linear) = linear {
                        names.insert(std::ptr::from_ref(linear) as usize,
                            (format!("{name}::{suffix}"), name.clone()));
                    }
                }
            }
        }
        let mut node = |linear: &mut VulkanLinear| -> Result<()> {
            if let Some((key, name)) = names.remove(&(std::ptr::from_ref(&*linear) as usize)) {
                visit(key, name, linear)?;
            }
            Ok(())
        };
        for (index, layer) in self.layers.iter_mut().enumerate() {
            if layer.falcon_h1.is_some() {
                for (_, linear) in layer.falcon_peft_linears_mut(index)? { node(linear)?; }
            }
            for linear in [&mut layer.c_attn, &mut layer.q_proj, &mut layer.k_proj,
                           &mut layer.v_proj, &mut layer.c_gate, &mut layer.attention_gate]
                           .into_iter().flatten() { node(linear)?; }
            for linear in [&mut layer.c_proj, &mut layer.c_fc, &mut layer.c_mlp_proj] { node(linear)?; }
            if let Some(mla) = &mut layer.mla_attention {
                for linear in [&mut mla.q_proj, &mut mla.q_a_proj, &mut mla.q_b_proj]
                    .into_iter().flatten() { node(linear)?; }
                node(&mut mla.kv_a_proj)?;
                node(&mut mla.kv_b_proj)?;
            }
            if let Some(delta) = &mut layer.qwen_gated_delta {
                match &mut delta.projection {
                    VulkanQwenGatedDeltaProjection::Qwen35 { qkv, z, b, a } => {
                        for linear in [qkv, z, b, a] { node(linear)?; }
                    }
                    VulkanQwenGatedDeltaProjection::Kimi { q, k, v, forget_a, forget_b, beta, gate, .. } => {
                        for linear in [q, k, v, forget_a, forget_b, beta, gate] { node(linear)?; }
                    }
                    _ => {}
                }
            }
            for moe in [&mut layer.moe, &mut layer.shortcut_moe].into_iter().flatten() {
                node(&mut moe.router)?;
                for linear in [&mut moe.shared_expert_gate, &mut moe.latent_down_proj,
                               &mut moe.latent_up_proj].into_iter().flatten() { node(linear)?; }
                for expert in moe.experts.iter_mut().chain(moe.owned_expert0.iter_mut())
                    .chain(moe.shared_expert.iter_mut()) {
                    if let Some(gate) = &mut expert.gate { node(gate)?; }
                    node(&mut expert.up)?;
                    node(&mut expert.down)?;
                }
            }
            if let Some(attention) = &mut layer.deepseek_v4_attention {
                for linear in [&mut attention.q_a_proj, &mut attention.q_b_proj,
                               &mut attention.kv_proj, &mut attention.o_b_proj] { node(linear)?; }
                if let Some(csa) = &mut attention.csa_compressor {
                    for linear in [&mut csa.compressor.kv_proj, &mut csa.compressor.gate_proj,
                        &mut csa.indexer.compressor.kv_proj, &mut csa.indexer.compressor.gate_proj,
                        &mut csa.indexer.q_b_proj, &mut csa.indexer.weights_proj] { node(linear)?; }
                }
                if let Some(hca) = &mut attention.hca_compressor {
                    node(&mut hca.kv_proj)?;
                    node(&mut hca.gate_proj)?;
                }
            }
            if let Some(ple) = &mut layer.qwen4_ple {
                node(&mut ple.key_proj)?;
                node(&mut ple.value_proj)?;
            }
            for residual in [&mut layer.qwen4_attn_hyper_connection, &mut layer.qwen4_mlp_hyper_connection]
                .into_iter().flatten() {
                node(&mut residual.input_mix_weight_down)?;
                node(&mut residual.input_mix_weight_up)?;
                if let Some(inject) = &mut residual.block_inject_weight { node(inject)?; }
            }
        }
        if let Some(mixer) = &mut self.qwen4_final_mixer {
            node(&mut mixer.input_mix_weight_down)?;
            node(&mut mixer.input_mix_weight_up)?;
        }
        if !names.is_empty() { bail!("PEFT replacement traversal missed native nodes: {:?}", names.values()); }
        Ok(())
    }

    fn for_each_verified_peft_norm_mut<F>(&mut self, mut visit: F) -> Result<()>
    where
        F: FnMut(String, &mut VulkanLayerNorm) -> Result<()>,
    {
        match self.config.architecture {
            VulkanTransformerArchitecture::Gpt2 => {
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    let p = format!("transformer.h.{index}");
                    visit(format!("{p}.ln_1"), &mut layer.ln1)?;
                    visit(format!("{p}.ln_2"), &mut layer.ln2)?;
                    if let Some(cross_attention) = layer.cross_attention.as_mut() {
                        visit(format!("{p}.ln_cross_attn"), &mut cross_attention.norm)?;
                    }
                }
                visit("transformer.ln_f".to_owned(), &mut self.final_norm)?;
            }
            VulkanTransformerArchitecture::Llama | VulkanTransformerArchitecture::Qwen2 => {
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    let p = format!("model.layers.{index}");
                    visit(format!("{p}.input_layernorm"), &mut layer.ln1)?;
                    visit(format!("{p}.post_attention_layernorm"), &mut layer.ln2)?;
                }
                visit("model.norm".to_owned(), &mut self.final_norm)?;
            }
            VulkanTransformerArchitecture::Mixtral | VulkanTransformerArchitecture::Qwen3Next => {
                for (name, norm) in self.peft_graph_norms_mut() { visit(name, norm)?; }
            }
            _ if self.config.architecture.peft_base_verified() => {
                for (name, norm) in self.peft_graph_norms_mut() { visit(name, norm)?; }
            },
            _ if self.peft_modules_to_save_norm_modules().is_empty() => {},
            architecture => bail!("PEFT norm replacement traversal missing for {}", architecture.model_type()),
        }
        Ok(())
    }

    pub(super) fn peft_graph_norms(&self) -> Vec<(String, &VulkanLayerNorm)> {
        let mut norms = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let p = self.peft_decoder_layer_prefix(index);
            if self.config.architecture == VulkanTransformerArchitecture::Qwen4Exp {
                for (name, residual) in [("attn_hyper_connection", &layer.qwen4_attn_hyper_connection),
                                         ("mlp_hyper_connection", &layer.qwen4_mlp_hyper_connection)] {
                    if let Some(residual) = residual { norms.push((format!("{p}.{name}.hc_norm"), &residual.norm)); }
                }
            } else {
                norms.push((format!("{p}.input_layernorm"), &layer.ln1));
                let second = match self.config.architecture {
                    VulkanTransformerArchitecture::Gemma3 | VulkanTransformerArchitecture::Gemma4 => "pre_feedforward_layernorm",
                    VulkanTransformerArchitecture::FalconH1 => "pre_ff_layernorm",
                    _ => "post_attention_layernorm",
                };
                norms.push((format!("{p}.{second}"), &layer.ln2));
                if let Some(norm) = &layer.post_attention_norm { norms.push((format!("{p}.post_attention_layernorm"), norm)); }
                if let Some(norm) = &layer.post_mlp_norm { norms.push((format!("{p}.post_feedforward_layernorm"), norm)); }
            }
        }
        if let Some(mixer) = &self.qwen4_final_mixer {
            norms.push(("model.hyper_connection_mixer.hc_norm".to_owned(), &mixer.norm));
        } else {
            let root = self.peft_decoder_layer_prefix(0);
            let root = root.strip_suffix(".layers.0").unwrap_or("model");
            norms.push((format!("{root}.norm"), &self.final_norm));
        }
        norms
    }

    pub(super) fn peft_graph_norms_mut(&mut self) -> Vec<(String, &mut VulkanLayerNorm)> {
        let root = self.peft_decoder_layer_prefix(0);
        let root = root.strip_suffix(".layers.0").unwrap_or("model").to_owned();
        let mut norms = Vec::new();
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("{root}.layers.{index}");
            if self.config.architecture == VulkanTransformerArchitecture::Qwen4Exp {
                for (name, residual) in [("attn_hyper_connection", &mut layer.qwen4_attn_hyper_connection),
                                         ("mlp_hyper_connection", &mut layer.qwen4_mlp_hyper_connection)] {
                    if let Some(residual) = residual { norms.push((format!("{p}.{name}.hc_norm"), &mut residual.norm)); }
                }
            } else {
                norms.push((format!("{p}.input_layernorm"), &mut layer.ln1));
                let second = match self.config.architecture {
                    VulkanTransformerArchitecture::Gemma3 | VulkanTransformerArchitecture::Gemma4 => "pre_feedforward_layernorm",
                    VulkanTransformerArchitecture::FalconH1 => "pre_ff_layernorm",
                    _ => "post_attention_layernorm",
                };
                norms.push((format!("{p}.{second}"), &mut layer.ln2));
                if let Some(norm) = &mut layer.post_attention_norm { norms.push((format!("{p}.post_attention_layernorm"), norm)); }
                if let Some(norm) = &mut layer.post_mlp_norm { norms.push((format!("{p}.post_feedforward_layernorm"), norm)); }
            }
        }
        if let Some(mixer) = &mut self.qwen4_final_mixer {
            norms.push(("model.hyper_connection_mixer.hc_norm".to_owned(), &mut mixer.norm));
        } else {
            norms.push((format!("{root}.norm"), &mut self.final_norm));
        }
        norms
    }

    /// Restore the union of canonical base modules before constructing or
    /// selecting another named adapter. This is the critical no-leakage step.
    pub(super) fn restore_peft_saved_module_base(&mut self) -> Result<()> {
        if self.lora_saved_module_base.is_empty() {
            self.lora_saved_embedding_module = None;
            return Ok(());
        }
        let base = self.lora_saved_module_base.clone();

        if let Some((_, parameter)) = base.embedding.as_ref() {
            self.shared_embedding = parameter.clone();
            self.token_embedding
                .rebind_shared_parameter(parameter.clone())?;
        }
        if let Some(lm_head) = base.lm_head.as_ref() {
            match lm_head {
                PeftSavedLmHeadState::Tied => self.lm_head = None,
                PeftSavedLmHeadState::Linear(state) => {
                    if self.lm_head.is_none() {
                        self.promote_tied_lm_head_for_peft_module_clone()?;
                    }
                    state.install(
                        self.lm_head
                            .as_mut()
                            .context("restored PEFT base lm_head disappeared")?,
                    );
                }
            }
        }
        self.for_each_verified_peft_linear_mut(|storage_key, _, linear| {
            if let Some(state) = base.linears.get(&storage_key) {
                state.install(linear);
            }
            Ok(())
        })?;
        self.for_each_verified_peft_norm_mut(|module, norm| {
            if let Some(state) = base.norms.get(&module) {
                state.install(norm);
            }
            Ok(())
        })?;
        self.lora_saved_embedding_module = None;
        Ok(())
    }

    /// Snapshot newly wrapped canonical modules before `configure_lora`
    /// replaces them. Existing entries are never overwritten, so the base map
    /// remains canonical across A/B/... adapter additions.
    pub(super) fn capture_peft_saved_module_base(
        &mut self,
        config: &VulkanTransformerLoraConfig,
        saved_embedding: Option<&str>,
    ) -> Result<()> {
        if config.modules_to_save.is_none() {
            return Ok(());
        }
        if !self.supports_verified_modules_to_save_bank() {
            return Ok(());
        }

        let mut base = std::mem::take(&mut self.lora_saved_module_base);
        let linear_result = self.for_each_verified_peft_linear_mut(|storage_key, module, linear| {
            if config.saves_module(&module) {
                base.linears
                    .entry(storage_key)
                    .or_insert_with(|| PeftSavedLinearState::frozen_base(linear));
            }
            Ok(())
        });
        if let Err(error) = linear_result {
            self.lora_saved_module_base = base;
            return Err(error);
        }
        let norm_result = self.for_each_verified_peft_norm_mut(|module, norm| {
            if config.saves_module(&module) {
                base.norms
                    .entry(module)
                    .or_insert_with(|| PeftSavedNormState::frozen_base(norm));
            }
            Ok(())
        });
        if let Err(error) = norm_result {
            self.lora_saved_module_base = base;
            return Err(error);
        }

        if (config.saves_module("lm_head") || saved_embedding.is_some()) && base.lm_head.is_none() {
            base.lm_head = Some(match self.lm_head.as_ref() {
                Some(lm_head) => {
                    PeftSavedLmHeadState::Linear(PeftSavedLinearState::frozen_base(lm_head))
                }
                None => PeftSavedLmHeadState::Tied,
            });
        }
        if let Some(module) = saved_embedding {
            if base.embedding.is_none() {
                base.embedding = Some((module.to_owned(), self.shared_embedding.clone()));
            }
        }
        self.lora_saved_module_base = base;
        Ok(())
    }

    /// Capture the live replacement handles after adapter construction. Clones
    /// preserve the adapter's Vulkan weights, gradient buffers and AdamW
    /// moments; later switching only swaps handles, never serializes them.
    pub(super) fn capture_peft_saved_module_adapter(
        &mut self,
        adapter_name: &str,
        config: &VulkanTransformerLoraConfig,
        saved_embedding: Option<&str>,
    ) -> Result<()> {
        if config.modules_to_save.is_none() || !self.supports_verified_modules_to_save_bank() {
            self.lora_saved_module_adapters
                .entry(adapter_name.to_owned())
                .or_default();
            return Ok(());
        }
        let mut state = PeftSavedModuleState::default();
        self.for_each_verified_peft_linear_mut(|storage_key, module, linear| {
            if config.saves_module(&module) {
                state
                    .linears
                    .insert(storage_key, PeftSavedLinearState::adapter(linear));
            }
            Ok(())
        })?;
        self.for_each_verified_peft_norm_mut(|module, norm| {
            if config.saves_module(&module) {
                state
                    .norms
                    .insert(module, PeftSavedNormState::adapter(norm));
            }
            Ok(())
        })?;
        if config.saves_module("lm_head") {
            state.lm_head = Some(PeftSavedLmHeadState::Linear(PeftSavedLinearState::adapter(
                self.lm_head
                    .as_ref()
                    .context("configured modules_to_save lm_head disappeared")?,
            )));
        }
        if let Some(module) = saved_embedding {
            state.embedding = Some((module.to_owned(), self.shared_embedding.clone()));
        }
        self.lora_saved_module_adapters
            .insert(adapter_name.to_owned(), state);
        Ok(())
    }

    /// Restore canonical base replacements and overlay one adapter's saved
    /// modules. This preserves exact A/B parameter and optimizer identities.
    pub(super) fn activate_peft_saved_module_adapter(&mut self, adapter_name: &str) -> Result<()> {
        self.restore_peft_saved_module_base()?;
        let Some(state) = self.lora_saved_module_adapters.get(adapter_name).cloned() else {
            return Ok(());
        };
        if state.is_empty() {
            return Ok(());
        }

        let config = self
            .lora_configs
            .get(adapter_name)
            .cloned()
            .with_context(|| format!("missing PEFT config for adapter {adapter_name:?}"))?;
        let ties_saved_embedding = state.embedding.is_some()
            && self.config.tie_word_embeddings
            && config.requests_saved_embedding_lm_head_tie();

        if state.embedding.is_some()
            && matches!(self.lora_saved_module_base.lm_head, Some(PeftSavedLmHeadState::Tied))
            && !ties_saved_embedding
            && self.lm_head.is_none()
        {
            // Clone the canonical tied output head before swapping in the
            // adapter-local input embedding, matching ModulesToSaveWrapper.
            self.promote_tied_lm_head_for_peft_module_clone()?;
            if let Some(lm_head) = self.lm_head.as_mut() {
                lm_head.freeze_base();
            }
        }
        if let Some((module, parameter)) = state.embedding.as_ref() {
            self.shared_embedding = parameter.clone();
            self.token_embedding
                .rebind_shared_parameter(parameter.clone())?;
            self.lora_saved_embedding_module = Some(module.clone());
            if ties_saved_embedding
                && matches!(self.lora_saved_module_base.lm_head, Some(PeftSavedLmHeadState::Tied))
            {
                self.lm_head = None;
            }
        }
        if let Some(PeftSavedLmHeadState::Linear(lm_head)) = state.lm_head.as_ref() {
            if self.lm_head.is_none() {
                self.promote_tied_lm_head_for_peft_module_clone()?;
            }
            lm_head.install(
                self.lm_head
                    .as_mut()
                    .context("selected PEFT lm_head replacement disappeared")?,
            );
        }
        self.for_each_verified_peft_linear_mut(|storage_key, _, linear| {
            if let Some(saved) = state.linears.get(&storage_key) {
                saved.install(linear);
            }
            Ok(())
        })?;
        self.for_each_verified_peft_norm_mut(|module, norm| {
            if let Some(saved) = state.norms.get(&module) {
                saved.install(norm);
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl VulkanTransformer {
    pub(super) fn peft_layer_linears(&self, index: usize) -> Result<Vec<(String, &VulkanLinear)>> {
        let layer = &self.layers[index];
        let modules = match self.config.architecture {
            VulkanTransformerArchitecture::FalconH1 => layer.falcon_peft_linears(index)?,
            VulkanTransformerArchitecture::KimiLinear => {
                let p = format!("model.layers.{index}");
                let mut modules = Vec::new();
                if let Some(gated_delta) = layer.qwen_gated_delta.as_ref() {
                    match &gated_delta.projection {
                        VulkanQwenGatedDeltaProjection::Kimi {
                            q,
                            k,
                            v,
                            forget_a,
                            forget_b,
                            beta,
                            gate,
                            ..
                        } => {
                            modules.extend([
                                (format!("{p}.self_attn.q_proj"), q),
                                (format!("{p}.self_attn.k_proj"), k),
                                (format!("{p}.self_attn.v_proj"), v),
                                (format!("{p}.self_attn.f_a_proj"), forget_a),
                                (format!("{p}.self_attn.f_b_proj"), forget_b),
                                (format!("{p}.self_attn.b_proj"), beta),
                                (format!("{p}.self_attn.g_proj"), gate),
                            ]);
                        }
                        _ => bail!("KimiLinear KDA layer uses a non-Kimi projection ABI"),
                    }
                    modules.push((format!("{p}.self_attn.o_proj"), &layer.c_proj));
                } else {
                    let mla = layer
                        .mla_attention
                        .as_ref()
                        .context("KimiLinear full-attention layer is missing MLA attention")?;
                    if let Some(q_proj) = mla.q_proj.as_ref() {
                        modules.push((format!("{p}.self_attn.q_proj"), q_proj));
                    } else {
                        modules.push((
                            format!("{p}.self_attn.q_a_proj"),
                            mla.q_a_proj
                                .as_ref()
                                .context("KimiLinear MLA is missing q_a_proj")?,
                        ));
                        modules.push((
                            format!("{p}.self_attn.q_b_proj"),
                            mla.q_b_proj
                                .as_ref()
                                .context("KimiLinear MLA is missing q_b_proj")?,
                        ));
                    }
                    modules.extend([
                        (format!("{p}.self_attn.kv_a_proj_with_mqa"), &mla.kv_a_proj),
                        (format!("{p}.self_attn.kv_b_proj"), &mla.kv_b_proj),
                        (format!("{p}.self_attn.o_proj"), &layer.c_proj),
                    ]);
                    if let Some(gate) = layer.attention_gate.as_ref() {
                        modules.push((format!("{p}.self_attn.g_proj"), gate));
                    }
                }

                if let Some(moe) = layer.moe.as_ref() {
                    let block = format!("{p}.block_sparse_moe");
                    if let Some(expert0) = moe.owned_expert0.as_ref() {
                        if let Some(gate) = expert0.gate.as_ref() {
                            modules.push((format!("{block}.experts.0.w1"), gate));
                        }
                        modules.push((format!("{block}.experts.0.w3"), &expert0.up));
                        modules.push((format!("{block}.experts.0.w2"), &expert0.down));
                    }
                    for (expert_offset, expert) in moe.experts.iter().enumerate() {
                        let expert_index = expert_offset + 1;
                        if let Some(gate) = expert.gate.as_ref() {
                            modules.push((format!("{block}.experts.{expert_index}.w1"), gate));
                        }
                        modules.push((format!("{block}.experts.{expert_index}.w3"), &expert.up));
                        modules.push((format!("{block}.experts.{expert_index}.w2"), &expert.down));
                    }
                    if let Some(shared) = moe.shared_expert.as_ref() {
                        if let Some(gate) = shared.gate.as_ref() {
                            modules.push((format!("{block}.shared_experts.gate_proj"), gate));
                        }
                        modules.push((format!("{block}.shared_experts.up_proj"), &shared.up));
                        modules.push((format!("{block}.shared_experts.down_proj"), &shared.down));
                    }
                    if let Some(down) = moe.latent_down_proj.as_ref() {
                        modules.push((format!("{block}.routed_expert_down_proj"), down));
                    }
                    if let Some(up) = moe.latent_up_proj.as_ref() {
                        modules.push((format!("{block}.routed_expert_up_proj"), up));
                    }
                } else {
                    modules.extend([
                        (
                            format!("{p}.mlp.gate_proj"),
                            layer
                                .c_gate
                                .as_ref()
                                .context("KimiLinear dense MLP is missing gate_proj")?,
                        ),
                        (format!("{p}.mlp.up_proj"), &layer.c_fc),
                        (format!("{p}.mlp.down_proj"), &layer.c_mlp_proj),
                    ]);
                }
                modules
            }
            VulkanTransformerArchitecture::T5Gemma => {
                bail!(
                    "T5Gemma PEFT export requires component-aware encoder/decoder module names"
                )
            }
            VulkanTransformerArchitecture::DeepseekV4 => {
                let p = format!("model.layers.{index}.self_attn");
                let attention = layer
                    .deepseek_v4_attention
                    .as_ref()
                    .context("DeepSeek-V4 layer is missing dedicated attention")?;
                let mut modules = vec![
                    (format!("{p}.q_a_proj"), &attention.q_a_proj),
                    (format!("{p}.q_b_proj"), &attention.q_b_proj),
                    (format!("{p}.kv_proj"), &attention.kv_proj),
                    (format!("{p}.o_b_proj"), &attention.o_b_proj),
                ];
                if let Some(csa) = attention.csa_compressor.as_ref() {
                    modules.extend([
                        (format!("{p}.compressor.kv_proj"), &csa.compressor.kv_proj),
                        (
                            format!("{p}.compressor.gate_proj"),
                            &csa.compressor.gate_proj,
                        ),
                        (
                            format!("{p}.compressor.indexer.kv_proj"),
                            &csa.indexer.compressor.kv_proj,
                        ),
                        (
                            format!("{p}.compressor.indexer.gate_proj"),
                            &csa.indexer.compressor.gate_proj,
                        ),
                        (
                            format!("{p}.compressor.indexer.q_b_proj"),
                            &csa.indexer.q_b_proj,
                        ),
                        (
                            format!("{p}.compressor.indexer.scorer.weights_proj"),
                            &csa.indexer.weights_proj,
                        ),
                    ]);
                } else if let Some(hca) = attention.hca_compressor.as_ref() {
                    modules.extend([
                        (format!("{p}.compressor.kv_proj"), &hca.kv_proj),
                        (format!("{p}.compressor.gate_proj"), &hca.gate_proj),
                    ]);
                }
                modules
            }
            VulkanTransformerArchitecture::Qwen3Next
            | VulkanTransformerArchitecture::Qwen35
            | VulkanTransformerArchitecture::Qwen35Moe
            | VulkanTransformerArchitecture::Qwen4Exp => {
                let p = format!("model.layers.{index}");
                let mut modules = Vec::new();
                if let Some(gated_delta) = layer.qwen_gated_delta.as_ref() {
                    match &gated_delta.projection {
                        VulkanQwenGatedDeltaProjection::Qwen3Next { qkvz, ba, .. } => {
                            modules.push((format!("{p}.linear_attn.in_proj_qkvz"), qkvz));
                            modules.push((format!("{p}.linear_attn.in_proj_ba"), ba));
                        }
                        VulkanQwenGatedDeltaProjection::Qwen35 { qkv, z, b, a } => {
                            modules.extend([
                                (format!("{p}.linear_attn.in_proj_qkv"), qkv),
                                (format!("{p}.linear_attn.in_proj_z"), z),
                                (format!("{p}.linear_attn.in_proj_b"), b),
                                (format!("{p}.linear_attn.in_proj_a"), a),
                            ]);
                        }
                        VulkanQwenGatedDeltaProjection::Glm5 { .. } => {
                            bail!("Qwen hybrid PEFT export encountered a GLM-5 KDA projection")
                        }
                        VulkanQwenGatedDeltaProjection::Kimi { .. } => {
                            bail!("Qwen hybrid PEFT export encountered a Kimi KDA projection")
                        }
                        VulkanQwenGatedDeltaProjection::OlmoHybrid { .. } => {
                            bail!(
                                "Qwen hybrid PEFT export encountered an OLMo Hybrid projection"
                            )
                        }
                    }
                    modules.push((format!("{p}.linear_attn.out_proj"), &layer.c_proj));
                } else {
                    modules.extend([
                        (
                            format!("{p}.self_attn.k_proj"),
                            layer.k_proj.as_ref().context(
                                "Qwen hybrid full-attention layer is missing k_proj",
                            )?,
                        ),
                        (
                            format!("{p}.self_attn.v_proj"),
                            layer.v_proj.as_ref().context(
                                "Qwen hybrid full-attention layer is missing v_proj",
                            )?,
                        ),
                        (format!("{p}.self_attn.o_proj"), &layer.c_proj),
                    ]);
                }
                if let Some(moe) = layer.moe.as_ref() {
                    if let Some(shared) = moe.shared_expert.as_ref() {
                        modules.extend([
                            (
                                format!("{p}.mlp.shared_expert.gate_proj"),
                                shared.gate.as_ref().context(
                                    "Qwen hybrid shared expert is missing gate_proj",
                                )?,
                            ),
                            (format!("{p}.mlp.shared_expert.up_proj"), &shared.up),
                            (format!("{p}.mlp.shared_expert.down_proj"), &shared.down),
                        ]);
                    }
                    if let Some(shared_gate) = moe.shared_expert_gate.as_ref() {
                        modules.push((format!("{p}.mlp.shared_expert_gate"), shared_gate));
                    }
                } else {
                    modules.extend([
                        (
                            format!("{p}.mlp.gate_proj"),
                            layer
                                .c_gate
                                .as_ref()
                                .context("Qwen hybrid dense MLP is missing gate_proj")?,
                        ),
                        (format!("{p}.mlp.up_proj"), &layer.c_fc),
                        (format!("{p}.mlp.down_proj"), &layer.c_mlp_proj),
                    ]);
                }
                if self.config.architecture == VulkanTransformerArchitecture::Qwen4Exp {
                    if let Some(ple) = layer.qwen4_ple.as_ref() {
                        modules.extend([
                            (format!("{p}.ple.key_proj"), &ple.key_proj),
                            (format!("{p}.ple.value_proj"), &ple.value_proj),
                        ]);
                    }
                    for (name, residual) in [
                        (
                            "attn_hyper_connection",
                            layer.qwen4_attn_hyper_connection.as_ref(),
                        ),
                        (
                            "mlp_hyper_connection",
                            layer.qwen4_mlp_hyper_connection.as_ref(),
                        ),
                    ] {
                        let Some(residual) = residual else {
                            continue;
                        };
                        let prefix = format!("{p}.{name}");
                        modules.extend([
                            (
                                format!("{prefix}.input_mix_weight_down"),
                                &residual.input_mix_weight_down,
                            ),
                            (
                                format!("{prefix}.input_mix_weight_up"),
                                &residual.input_mix_weight_up,
                            ),
                        ]);
                        if let Some(inject) = residual.block_inject_weight.as_ref() {
                            modules.push((format!("{prefix}.block_inject_weight"), inject));
                        }
                    }
                    if index + 1 == self.layers.len() {
                        if let Some(mixer) = self.qwen4_final_mixer.as_ref() {
                            modules.extend([
                                (
                                    "model.hyper_connection_mixer.input_mix_weight_down"
                                        .to_owned(),
                                    &mixer.input_mix_weight_down,
                                ),
                                (
                                    "model.hyper_connection_mixer.input_mix_weight_up"
                                        .to_owned(),
                                    &mixer.input_mix_weight_up,
                                ),
                            ]);
                        }
                    }
                }
                modules
            }
            VulkanTransformerArchitecture::Glm5Next => {
                let p = format!("model.layers.{index}");
                let mut modules = Vec::new();
                if let Some(gated_delta) = layer.qwen_gated_delta.as_ref() {
                    match &gated_delta.projection {
                        VulkanQwenGatedDeltaProjection::Glm5 {
                            q,
                            k,
                            v,
                            forget_a,
                            forget_b,
                            beta,
                            gate_a,
                            gate_b,
                            ..
                        } => {
                            modules.extend([
                                (format!("{p}.linear_attn.q_proj"), q),
                                (format!("{p}.linear_attn.k_proj"), k),
                                (format!("{p}.linear_attn.v_proj"), v),
                                (
                                    format!("{p}.linear_attn.forget_gate.f_a_proj"),
                                    forget_a,
                                ),
                                (
                                    format!("{p}.linear_attn.forget_gate.f_b_proj"),
                                    forget_b,
                                ),
                                (format!("{p}.linear_attn.b_proj"), beta),
                                (format!("{p}.linear_attn.g_a_proj"), gate_a),
                                (format!("{p}.linear_attn.g_b_proj"), gate_b),
                            ]);
                        }
                        _ => bail!(
                            "GLM-5 Next linear-attention layer uses a non-GLM KDA projection ABI"
                        ),
                    }
                    modules.push((format!("{p}.linear_attn.o_proj"), &layer.c_proj));
                } else {
                    let mla = layer
                        .mla_attention
                        .as_ref()
                        .context("GLM-5 Next DSA layer is missing MLA attention")?;
                    if let Some(q_proj) = mla.q_proj.as_ref() {
                        modules.push((format!("{p}.self_attn.q_proj"), q_proj));
                    } else {
                        modules.push((
                            format!("{p}.self_attn.q_a_proj"),
                            mla.q_a_proj
                                .as_ref()
                                .context("GLM-5 Next MLA is missing q_a_proj")?,
                        ));
                        modules.push((
                            format!("{p}.self_attn.q_b_proj"),
                            mla.q_b_proj
                                .as_ref()
                                .context("GLM-5 Next MLA is missing q_b_proj")?,
                        ));
                    }
                    modules.push((format!("{p}.self_attn.kv_a_proj_with_mqa"), &mla.kv_a_proj));
                    modules.push((format!("{p}.self_attn.kv_b_proj"), &mla.kv_b_proj));
                    modules.push((format!("{p}.self_attn.o_proj"), &layer.c_proj));
                    if let Some(indexer) = layer.dsa_indexer.as_ref() {
                        modules.extend([
                            (format!("{p}.self_attn.indexer.wq_b"), &indexer.wq_b),
                            (format!("{p}.self_attn.indexer.wk"), &indexer.wk),
                            (
                                format!("{p}.self_attn.indexer.weights_proj"),
                                &indexer.weights_proj,
                            ),
                        ]);
                    }
                }
                if let Some(moe) = layer.moe.as_ref() {
                    if let Some(shared) = moe.shared_expert.as_ref() {
                        modules.extend([
                            (
                                format!("{p}.mlp.shared_experts.gate_proj"),
                                shared.gate.as_ref().context(
                                    "GLM-5 Next shared experts are missing gate_proj",
                                )?,
                            ),
                            (format!("{p}.mlp.shared_experts.up_proj"), &shared.up),
                            (format!("{p}.mlp.shared_experts.down_proj"), &shared.down),
                        ]);
                    }
                } else {
                    modules.extend([
                        (
                            format!("{p}.mlp.gate_proj"),
                            layer
                                .c_gate
                                .as_ref()
                                .context("GLM-5 Next dense MLP is missing gate_proj")?,
                        ),
                        (format!("{p}.mlp.up_proj"), &layer.c_fc),
                        (format!("{p}.mlp.down_proj"), &layer.c_mlp_proj),
                    ]);
                }
                modules
            }
            VulkanTransformerArchitecture::OlmoHybrid => {
                let p = format!("model.layers.{index}");
                let mut modules = Vec::new();
                if let Some(gated_delta) = layer.qwen_gated_delta.as_ref() {
                    match &gated_delta.projection {
                        VulkanQwenGatedDeltaProjection::OlmoHybrid { q, k, v, z, b, a, .. } => {
                            modules.extend([
                                (format!("{p}.linear_attn.q_proj"), q),
                                (format!("{p}.linear_attn.k_proj"), k),
                                (format!("{p}.linear_attn.v_proj"), v),
                                (format!("{p}.linear_attn.g_proj"), z),
                                (format!("{p}.linear_attn.b_proj"), b),
                                (format!("{p}.linear_attn.a_proj"), a),
                            ]);
                        }
                        _ => bail!(
                            "OLMo Hybrid linear-attention layer uses an unexpected Gated DeltaNet projection ABI"
                        ),
                    }
                    modules.push((format!("{p}.linear_attn.o_proj"), &layer.c_proj));
                } else {
                    modules.extend([
                        (
                            format!("{p}.self_attn.q_proj"),
                            layer.q_proj.as_ref().context(
                                "OLMo Hybrid full-attention layer is missing q_proj",
                            )?,
                        ),
                        (
                            format!("{p}.self_attn.k_proj"),
                            layer.k_proj.as_ref().context(
                                "OLMo Hybrid full-attention layer is missing k_proj",
                            )?,
                        ),
                        (
                            format!("{p}.self_attn.v_proj"),
                            layer.v_proj.as_ref().context(
                                "OLMo Hybrid full-attention layer is missing v_proj",
                            )?,
                        ),
                        (format!("{p}.self_attn.o_proj"), &layer.c_proj),
                    ]);
                }
                modules.extend([
                    (
                        format!("{p}.mlp.gate_proj"),
                        layer
                            .c_gate
                            .as_ref()
                            .context("OLMo Hybrid MLP is missing gate_proj")?,
                    ),
                    (format!("{p}.mlp.up_proj"), &layer.c_fc),
                    (format!("{p}.mlp.down_proj"), &layer.c_mlp_proj),
                ]);
                modules
            }
            VulkanTransformerArchitecture::OpenAiGpt
            | VulkanTransformerArchitecture::Gpt2
            | VulkanTransformerArchitecture::GptSw3 => {
                vec![
                    (
                        format!("transformer.h.{index}.attn.c_attn"),
                        layer
                            .c_attn
                            .as_ref()
                            .context("OpenAI GPT/GPT-2 layer is missing c_attn")?,
                    ),
                    (format!("transformer.h.{index}.attn.c_proj"), &layer.c_proj),
                    (format!("transformer.h.{index}.mlp.c_fc"), &layer.c_fc),
                    (
                        format!("transformer.h.{index}.mlp.c_proj"),
                        &layer.c_mlp_proj,
                    ),
                ]
            }
            VulkanTransformerArchitecture::GptBigCode => vec![
                (
                    format!("transformer.h.{index}.attn.c_attn"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("GPT-BigCode layer is missing c_attn")?,
                ),
                (format!("transformer.h.{index}.attn.c_proj"), &layer.c_proj),
                (format!("transformer.h.{index}.mlp.c_fc"), &layer.c_fc),
                (
                    format!("transformer.h.{index}.mlp.c_proj"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::Falcon | VulkanTransformerArchitecture::Bloom => {
                vec![
                    (
                        format!("transformer.h.{index}.self_attention.query_key_value"),
                        layer
                            .c_attn
                            .as_ref()
                            .context("Falcon/BLOOM layer is missing query_key_value")?,
                    ),
                    (
                        format!("transformer.h.{index}.self_attention.dense"),
                        &layer.c_proj,
                    ),
                    (
                        format!("transformer.h.{index}.mlp.dense_h_to_4h"),
                        &layer.c_fc,
                    ),
                    (
                        format!("transformer.h.{index}.mlp.dense_4h_to_h"),
                        &layer.c_mlp_proj,
                    ),
                ]
            }
            VulkanTransformerArchitecture::Mpt => vec![
                (
                    format!("transformer.blocks.{index}.attn.Wqkv"),
                    layer.c_attn.as_ref().context("MPT layer is missing Wqkv")?,
                ),
                (
                    format!("transformer.blocks.{index}.attn.out_proj"),
                    &layer.c_proj,
                ),
                (
                    format!("transformer.blocks.{index}.ffn.up_proj"),
                    &layer.c_fc,
                ),
                (
                    format!("transformer.blocks.{index}.ffn.down_proj"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::Dbrx => vec![
                (
                    format!("transformer.blocks.{index}.norm_attn_norm.attn.Wqkv"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("DBRX layer is missing Wqkv")?,
                ),
                (
                    format!("transformer.blocks.{index}.norm_attn_norm.attn.out_proj"),
                    &layer.c_proj,
                ),
            ],
            VulkanTransformerArchitecture::ModernBert => vec![
                (
                    format!("model.layers.{index}.attn.Wqkv"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("ModernBERT layer is missing Wqkv")?,
                ),
                (format!("model.layers.{index}.attn.Wo"), &layer.c_proj),
                (format!("model.layers.{index}.mlp.Wo"), &layer.c_mlp_proj),
            ],
            VulkanTransformerArchitecture::ModernBertDecoder => vec![
                (
                    format!("model.layers.{index}.attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("ModernBERT Decoder layer is missing q_proj")?,
                ),
                (
                    format!("model.layers.{index}.attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("ModernBERT Decoder layer is missing k_proj")?,
                ),
                (
                    format!("model.layers.{index}.attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("ModernBERT Decoder layer is missing v_proj")?,
                ),
                (format!("model.layers.{index}.attn.Wo"), &layer.c_proj),
                (format!("model.layers.{index}.mlp.Wo"), &layer.c_mlp_proj),
            ],
            VulkanTransformerArchitecture::Esmc => {
                let p = format!("esmc.layers.{index}");
                vec![
                    (
                        format!("{p}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("ESMC layer is missing q_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("ESMC layer is missing k_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("ESMC layer is missing v_proj")?,
                    ),
                    (format!("{p}.self_attn.o_proj"), &layer.c_proj),
                    (
                        format!("{p}.mlp.gate_proj"),
                        layer
                            .c_gate
                            .as_ref()
                            .context("ESMC layer is missing gate_proj")?,
                    ),
                    (format!("{p}.mlp.up_proj"), &layer.c_fc),
                    (format!("{p}.mlp.down_proj"), &layer.c_mlp_proj),
                ]
            }
            VulkanTransformerArchitecture::NomicBert => {
                let p = format!("nomic_bert.layers.{index}");
                vec![
                    (
                        format!("{p}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("Nomic BERT layer is missing q_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("Nomic BERT layer is missing k_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("Nomic BERT layer is missing v_proj")?,
                    ),
                    (format!("{p}.self_attn.o_proj"), &layer.c_proj),
                    (
                        format!("{p}.mlp.gate_proj"),
                        layer
                            .c_gate
                            .as_ref()
                            .context("Nomic BERT layer is missing gate_proj")?,
                    ),
                    (format!("{p}.mlp.up_proj"), &layer.c_fc),
                    (format!("{p}.mlp.down_proj"), &layer.c_mlp_proj),
                ]
            }
            VulkanTransformerArchitecture::JinaEmbeddingsV3 => {
                let p = format!("roberta.layers.{index}");
                vec![
                    (
                        format!("{p}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("Jina Embeddings v3 layer is missing q_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("Jina Embeddings v3 layer is missing k_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("Jina Embeddings v3 layer is missing v_proj")?,
                    ),
                    (format!("{p}.self_attn.o_proj"), &layer.c_proj),
                    (format!("{p}.mlp.fc1"), &layer.c_fc),
                    (format!("{p}.mlp.fc2"), &layer.c_mlp_proj),
                ]
            }
            VulkanTransformerArchitecture::Bert
            | VulkanTransformerArchitecture::BigBird
            | VulkanTransformerArchitecture::RoCBert
            | VulkanTransformerArchitecture::RemBert
            | VulkanTransformerArchitecture::Electra
            | VulkanTransformerArchitecture::Ernie
            | VulkanTransformerArchitecture::BertGeneration
            | VulkanTransformerArchitecture::MegatronBert
            | VulkanTransformerArchitecture::RoFormer
            | VulkanTransformerArchitecture::Roberta
            | VulkanTransformerArchitecture::RobertaPreLayerNorm
            | VulkanTransformerArchitecture::Camembert
            | VulkanTransformerArchitecture::XlmRoberta
            | VulkanTransformerArchitecture::XlmRobertaXL
            | VulkanTransformerArchitecture::Data2VecText
            | VulkanTransformerArchitecture::Esm => {
                let base = self
                    .config
                    .architecture
                    .bert_family_base_prefix()
                    .expect("BERT-family LoRA route must have a checkpoint prefix");
                let p = format!("{base}.encoder.layer.{index}");
                let mut modules = vec![
                    (
                        format!("{p}.attention.self.query"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("BERT-family layer is missing query projection")?,
                    ),
                    (
                        format!("{p}.attention.self.key"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("BERT-family layer is missing key projection")?,
                    ),
                    (
                        format!("{p}.attention.self.value"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("BERT-family layer is missing value projection")?,
                    ),
                    (format!("{p}.attention.output.dense"), &layer.c_proj),
                    (format!("{p}.intermediate.dense"), &layer.c_fc),
                    (format!("{p}.output.dense"), &layer.c_mlp_proj),
                ];
                if let Some(cross_attention) = layer.cross_attention.as_ref() {
                    modules.extend([
                        (
                            format!("{p}.crossattention.self.query"),
                            &cross_attention.q_proj,
                        ),
                        (
                            format!("{p}.crossattention.self.key"),
                            &cross_attention.k_proj,
                        ),
                        (
                            format!("{p}.crossattention.self.value"),
                            &cross_attention.v_proj,
                        ),
                        (
                            format!("{p}.crossattention.output.dense"),
                            &cross_attention.out_proj,
                        ),
                    ]);
                }
                modules
            }
            VulkanTransformerArchitecture::DistilBert => {
                let p = format!("distilbert.transformer.layer.{index}");
                vec![
                    (
                        format!("{p}.attention.q_lin"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("DistilBERT layer is missing q_lin")?,
                    ),
                    (
                        format!("{p}.attention.k_lin"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("DistilBERT layer is missing k_lin")?,
                    ),
                    (
                        format!("{p}.attention.v_lin"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("DistilBERT layer is missing v_lin")?,
                    ),
                    (format!("{p}.attention.out_lin"), &layer.c_proj),
                    (format!("{p}.ffn.lin1"), &layer.c_fc),
                    (format!("{p}.ffn.lin2"), &layer.c_mlp_proj),
                ]
            }
            VulkanTransformerArchitecture::Xlm | VulkanTransformerArchitecture::Flaubert => {
                let attention = format!("transformer.attentions.{index}");
                let ffn = format!("transformer.ffns.{index}");
                vec![
                    (
                        format!("{attention}.q_lin"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("XLM-family layer is missing q_lin")?,
                    ),
                    (
                        format!("{attention}.k_lin"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("XLM-family layer is missing k_lin")?,
                    ),
                    (
                        format!("{attention}.v_lin"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("XLM-family layer is missing v_lin")?,
                    ),
                    (format!("{attention}.out_lin"), &layer.c_proj),
                    (format!("{ffn}.lin1"), &layer.c_fc),
                    (format!("{ffn}.lin2"), &layer.c_mlp_proj),
                ]
            }
            VulkanTransformerArchitecture::CodeGen => vec![
                (
                    format!("transformer.h.{index}.attn.qkv_proj"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("CodeGen layer is missing qkv_proj")?,
                ),
                (
                    format!("transformer.h.{index}.attn.out_proj"),
                    &layer.c_proj,
                ),
                (format!("transformer.h.{index}.mlp.fc_in"), &layer.c_fc),
                (
                    format!("transformer.h.{index}.mlp.fc_out"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::Ctrl => {
                let attention = format!("transformer.h.{index}.multi_head_attention");
                vec![
                    (
                        format!("{attention}.Wq"),
                        layer.q_proj.as_ref().context("CTRL layer is missing Wq")?,
                    ),
                    (
                        format!("{attention}.Wk"),
                        layer.k_proj.as_ref().context("CTRL layer is missing Wk")?,
                    ),
                    (
                        format!("{attention}.Wv"),
                        layer.v_proj.as_ref().context("CTRL layer is missing Wv")?,
                    ),
                    (format!("{attention}.dense"), &layer.c_proj),
                    (format!("transformer.h.{index}.ffn.0"), &layer.c_fc),
                    (format!("transformer.h.{index}.ffn.2"), &layer.c_mlp_proj),
                ]
            }
            VulkanTransformerArchitecture::GptNeo => {
                let attention = format!("transformer.h.{index}.attn.attention");
                vec![
                    (
                        format!("{attention}.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("GPT-Neo layer is missing q_proj")?,
                    ),
                    (
                        format!("{attention}.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("GPT-Neo layer is missing k_proj")?,
                    ),
                    (
                        format!("{attention}.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("GPT-Neo layer is missing v_proj")?,
                    ),
                    (format!("{attention}.out_proj"), &layer.c_proj),
                    (format!("transformer.h.{index}.mlp.c_fc"), &layer.c_fc),
                    (
                        format!("transformer.h.{index}.mlp.c_proj"),
                        &layer.c_mlp_proj,
                    ),
                ]
            }
            VulkanTransformerArchitecture::GptNeoX => vec![
                (
                    format!("gpt_neox.layers.{index}.attention.query_key_value"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("GPT-NeoX layer is missing c_attn")?,
                ),
                (
                    format!("gpt_neox.layers.{index}.attention.dense"),
                    &layer.c_proj,
                ),
                (
                    format!("gpt_neox.layers.{index}.mlp.dense_h_to_4h"),
                    &layer.c_fc,
                ),
                (
                    format!("gpt_neox.layers.{index}.mlp.dense_4h_to_h"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::GptNeoXJapanese => vec![
                (
                    format!("gpt_neox_japanese.layers.{index}.attention.query_key_value"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("GPT-NeoX-Japanese layer is missing c_attn")?,
                ),
                (
                    format!("gpt_neox_japanese.layers.{index}.attention.dense"),
                    &layer.c_proj,
                ),
                (
                    format!("gpt_neox_japanese.layers.{index}.mlp.dense_h_to_4h"),
                    &layer.c_fc,
                ),
                (
                    format!("gpt_neox_japanese.layers.{index}.mlp.dense_4h_to_h"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::Persimmon => vec![
                (
                    format!("model.layers.{index}.self_attn.query_key_value"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("Persimmon layer is missing query_key_value")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.dense"),
                    &layer.c_proj,
                ),
                (
                    format!("model.layers.{index}.mlp.dense_h_to_4h"),
                    &layer.c_fc,
                ),
                (
                    format!("model.layers.{index}.mlp.dense_4h_to_h"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::GptJ => vec![
                (
                    format!("transformer.h.{index}.attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("GPT-J layer is missing q_proj")?,
                ),
                (
                    format!("transformer.h.{index}.attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("GPT-J layer is missing k_proj")?,
                ),
                (
                    format!("transformer.h.{index}.attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("GPT-J layer is missing v_proj")?,
                ),
                (
                    format!("transformer.h.{index}.attn.out_proj"),
                    &layer.c_proj,
                ),
                (format!("transformer.h.{index}.mlp.fc_in"), &layer.c_fc),
                (
                    format!("transformer.h.{index}.mlp.fc_out"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::Phi => vec![
                (
                    format!("model.layers.{index}.self_attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("Phi layer is missing q_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("Phi layer is missing k_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("Phi layer is missing v_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.dense"),
                    &layer.c_proj,
                ),
                (format!("model.layers.{index}.mlp.fc1"), &layer.c_fc),
                (format!("model.layers.{index}.mlp.fc2"), &layer.c_mlp_proj),
            ],
            VulkanTransformerArchitecture::Starcoder2 => vec![
                (
                    format!("model.layers.{index}.self_attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("StarCoder2 layer is missing q_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("StarCoder2 layer is missing k_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("StarCoder2 layer is missing v_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj"),
                    &layer.c_proj,
                ),
                (format!("model.layers.{index}.mlp.c_fc"), &layer.c_fc),
                (
                    format!("model.layers.{index}.mlp.c_proj"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::T5
            | VulkanTransformerArchitecture::MT5
            | VulkanTransformerArchitecture::UMT5
            | VulkanTransformerArchitecture::SwitchTransformers => {
                let self_attention = format!("decoder.block.{index}.layer.0.SelfAttention");
                let mut modules = vec![
                    (
                        format!("{self_attention}.q"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("T5 layer is missing self-attention q")?,
                    ),
                    (
                        format!("{self_attention}.k"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("T5 layer is missing self-attention k")?,
                    ),
                    (
                        format!("{self_attention}.v"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("T5 layer is missing self-attention v")?,
                    ),
                    (format!("{self_attention}.o"), &layer.c_proj),
                ];
                let cross_attention = layer
                    .cross_attention
                    .as_ref()
                    .context("T5 decoder layer is missing cross-attention")?;
                let cross_prefix = format!("decoder.block.{index}.layer.1.EncDecAttention");
                modules.extend([
                    (format!("{cross_prefix}.q"), &cross_attention.q_proj),
                    (format!("{cross_prefix}.k"), &cross_attention.k_proj),
                    (format!("{cross_prefix}.v"), &cross_attention.v_proj),
                    (format!("{cross_prefix}.o"), &cross_attention.out_proj),
                ]);
                if self.config.architecture == VulkanTransformerArchitecture::SwitchTransformers
                {
                    let mlp = format!("decoder.block.{index}.layer.2.mlp");
                    if self.config.layer_uses_moe(index) {
                        let moe = layer.moe.as_ref().context(
                            "Switch Transformers sparse layer is missing MoE parameters",
                        )?;
                        modules.push((format!("{mlp}.router.classifier"), &moe.router));
                        modules.push((format!("{mlp}.experts.expert_0.wi"), &layer.c_fc));
                        modules.push((format!("{mlp}.experts.expert_0.wo"), &layer.c_mlp_proj));
                        for (expert_offset, expert) in moe.experts.iter().enumerate() {
                            let expert_index = expert_offset + 1;
                            modules.push((
                                format!("{mlp}.experts.expert_{expert_index}.wi"),
                                &expert.up,
                            ));
                            modules.push((
                                format!("{mlp}.experts.expert_{expert_index}.wo"),
                                &expert.down,
                            ));
                        }
                    } else {
                        modules.push((format!("{mlp}.wi"), &layer.c_fc));
                        modules.push((format!("{mlp}.wo"), &layer.c_mlp_proj));
                    }
                } else {
                    let dense = format!("decoder.block.{index}.layer.2.DenseReluDense");
                    if let Some(gate) = layer.c_gate.as_ref() {
                        modules.push((format!("{dense}.wi_0"), gate));
                        modules.push((format!("{dense}.wi_1"), &layer.c_fc));
                    } else {
                        modules.push((format!("{dense}.wi"), &layer.c_fc));
                    }
                    modules.push((format!("{dense}.wo"), &layer.c_mlp_proj));
                }
                modules
            }
            VulkanTransformerArchitecture::TrOCR
            | VulkanTransformerArchitecture::Bart
            | VulkanTransformerArchitecture::MBart
            | VulkanTransformerArchitecture::Marian
            | VulkanTransformerArchitecture::Fsmt
            | VulkanTransformerArchitecture::PLBart
            | VulkanTransformerArchitecture::Mvp
            | VulkanTransformerArchitecture::Blenderbot
            | VulkanTransformerArchitecture::BlenderbotSmall
            | VulkanTransformerArchitecture::Pegasus
            | VulkanTransformerArchitecture::M2M100
            | VulkanTransformerArchitecture::BigBirdPegasus
            | VulkanTransformerArchitecture::PegasusX => {
                let p = format!("model.decoder.layers.{index}");
                let mut modules = vec![
                    (
                        format!("{p}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("BART layer is missing q_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("BART layer is missing k_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("BART layer is missing v_proj")?,
                    ),
                    (format!("{p}.self_attn.out_proj"), &layer.c_proj),
                    (format!("{p}.fc1"), &layer.c_fc),
                    (format!("{p}.fc2"), &layer.c_mlp_proj),
                ];
                if let Some(cross_attention) = layer.cross_attention.as_ref() {
                    modules.extend([
                        (format!("{p}.encoder_attn.q_proj"), &cross_attention.q_proj),
                        (format!("{p}.encoder_attn.k_proj"), &cross_attention.k_proj),
                        (format!("{p}.encoder_attn.v_proj"), &cross_attention.v_proj),
                        (
                            format!("{p}.encoder_attn.out_proj"),
                            &cross_attention.out_proj,
                        ),
                    ]);
                }
                modules
            }
            VulkanTransformerArchitecture::Opt => vec![
                (
                    format!("model.decoder.layers.{index}.self_attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("OPT layer is missing q_proj")?,
                ),
                (
                    format!("model.decoder.layers.{index}.self_attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("OPT layer is missing k_proj")?,
                ),
                (
                    format!("model.decoder.layers.{index}.self_attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("OPT layer is missing v_proj")?,
                ),
                (
                    format!("model.decoder.layers.{index}.self_attn.out_proj"),
                    &layer.c_proj,
                ),
                (format!("model.decoder.layers.{index}.fc1"), &layer.c_fc),
                (
                    format!("model.decoder.layers.{index}.fc2"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::BioGpt => vec![
                (
                    format!("biogpt.layers.{index}.self_attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("BioGPT layer is missing q_proj")?,
                ),
                (
                    format!("biogpt.layers.{index}.self_attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("BioGPT layer is missing k_proj")?,
                ),
                (
                    format!("biogpt.layers.{index}.self_attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("BioGPT layer is missing v_proj")?,
                ),
                (
                    format!("biogpt.layers.{index}.self_attn.out_proj"),
                    &layer.c_proj,
                ),
                (format!("biogpt.layers.{index}.fc1"), &layer.c_fc),
                (format!("biogpt.layers.{index}.fc2"), &layer.c_mlp_proj),
            ],
            VulkanTransformerArchitecture::Xglm => {
                let p = format!("model.layers.{index}");
                let mut modules = vec![
                    (
                        format!("{p}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("XGLM layer is missing q_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("XGLM layer is missing k_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("XGLM layer is missing v_proj")?,
                    ),
                    (format!("{p}.self_attn.out_proj"), &layer.c_proj),
                    (format!("{p}.fc1"), &layer.c_fc),
                    (format!("{p}.fc2"), &layer.c_mlp_proj),
                ];
                if let Some(cross_attention) = layer.cross_attention.as_ref() {
                    modules.extend([
                        (format!("{p}.encoder_attn.q_proj"), &cross_attention.q_proj),
                        (format!("{p}.encoder_attn.k_proj"), &cross_attention.k_proj),
                        (format!("{p}.encoder_attn.v_proj"), &cross_attention.v_proj),
                        (
                            format!("{p}.encoder_attn.out_proj"),
                            &cross_attention.out_proj,
                        ),
                    ]);
                }
                modules
            }
            VulkanTransformerArchitecture::Glm | VulkanTransformerArchitecture::Glm4 => vec![
                (
                    format!("model.layers.{index}.self_attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("GLM layer is missing q_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("GLM layer is missing k_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("GLM layer is missing v_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj"),
                    &layer.c_proj,
                ),
                (
                    format!("model.layers.{index}.mlp.down_proj"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::Afmoe => vec![
                (
                    format!("model.layers.{index}.self_attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("AFMoE layer is missing q_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("AFMoE layer is missing k_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("AFMoE layer is missing v_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj"),
                    &layer.c_proj,
                ),
                (
                    format!("model.layers.{index}.self_attn.gate_proj"),
                    layer
                        .attention_gate
                        .as_ref()
                        .context("AFMoE layer is missing self-attention gate_proj")?,
                ),
            ],
            VulkanTransformerArchitecture::Laguna => {
                let mut modules = vec![
                    (
                        format!("model.layers.{index}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("Laguna layer is missing q_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("Laguna layer is missing k_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("Laguna layer is missing v_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.o_proj"),
                        &layer.c_proj,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.g_proj"),
                        layer
                            .attention_gate
                            .as_ref()
                            .context("Laguna layer is missing self-attention g_proj")?,
                    ),
                ];
                if !self.config.layer_uses_moe(index) {
                    modules.push((
                        format!("model.layers.{index}.mlp.gate_proj"),
                        layer
                            .c_gate
                            .as_ref()
                            .context("Laguna dense layer is missing gate_proj")?,
                    ));
                    modules.push((format!("model.layers.{index}.mlp.up_proj"), &layer.c_fc));
                    modules.push((
                        format!("model.layers.{index}.mlp.down_proj"),
                        &layer.c_mlp_proj,
                    ));
                }
                modules
            }
            VulkanTransformerArchitecture::Llama4Text => {
                let mut modules = vec![
                    (
                        format!("model.layers.{index}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("Llama 4 layer is missing q_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("Llama 4 layer is missing k_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("Llama 4 layer is missing v_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.o_proj"),
                        &layer.c_proj,
                    ),
                ];
                if !self.config.layer_uses_moe(index) {
                    modules.push((
                        format!("model.layers.{index}.feed_forward.gate_proj"),
                        layer
                            .c_gate
                            .as_ref()
                            .context("Llama 4 dense layer is missing gate_proj")?,
                    ));
                    modules.push((
                        format!("model.layers.{index}.feed_forward.up_proj"),
                        &layer.c_fc,
                    ));
                    modules.push((
                        format!("model.layers.{index}.feed_forward.down_proj"),
                        &layer.c_mlp_proj,
                    ));
                }
                modules
            }
            VulkanTransformerArchitecture::MiMoV2Flash => {
                let mut modules = vec![
                    (
                        format!("model.layers.{index}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("MiMo-V2-Flash layer is missing q_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("MiMo-V2-Flash layer is missing k_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("MiMo-V2-Flash layer is missing v_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.o_proj"),
                        &layer.c_proj,
                    ),
                ];
                if !self.config.layer_uses_moe(index) {
                    modules.push((
                        format!("model.layers.{index}.mlp.gate_proj"),
                        layer
                            .c_gate
                            .as_ref()
                            .context("MiMo-V2-Flash dense layer is missing gate_proj")?,
                    ));
                    modules.push((format!("model.layers.{index}.mlp.up_proj"), &layer.c_fc));
                    modules.push((
                        format!("model.layers.{index}.mlp.down_proj"),
                        &layer.c_mlp_proj,
                    ));
                }
                modules
            }
            VulkanTransformerArchitecture::LongCatFlash => {
                let logical_layer = index / 2;
                let sublayer = index % 2;
                let p = format!("model.layers.{logical_layer}");
                let attention = format!("{p}.self_attn.{sublayer}");
                let mla = layer
                    .mla_attention
                    .as_ref()
                    .context("LongCat-Flash layer is missing MLA attention")?;
                let mut modules = Vec::new();
                if let Some(q_proj) = mla.q_proj.as_ref() {
                    modules.push((format!("{attention}.q_proj"), q_proj));
                } else {
                    modules.push((
                        format!("{attention}.q_a_proj"),
                        mla.q_a_proj
                            .as_ref()
                            .context("LongCat-Flash MLA is missing q_a_proj")?,
                    ));
                    modules.push((
                        format!("{attention}.q_b_proj"),
                        mla.q_b_proj
                            .as_ref()
                            .context("LongCat-Flash MLA is missing q_b_proj")?,
                    ));
                }
                modules.push((format!("{attention}.kv_a_proj_with_mqa"), &mla.kv_a_proj));
                modules.push((format!("{attention}.kv_b_proj"), &mla.kv_b_proj));
                modules.push((format!("{attention}.o_proj"), &layer.c_proj));
                let mlp = format!("{p}.mlps.{sublayer}");
                modules.push((
                    format!("{mlp}.gate_proj"),
                    layer
                        .c_gate
                        .as_ref()
                        .context("LongCat-Flash dense MLP is missing gate_proj")?,
                ));
                modules.push((format!("{mlp}.up_proj"), &layer.c_fc));
                modules.push((format!("{mlp}.down_proj"), &layer.c_mlp_proj));
                modules
            }
            VulkanTransformerArchitecture::DeepseekV2
            | VulkanTransformerArchitecture::MiniCpm3
            | VulkanTransformerArchitecture::DeepseekV3
            | VulkanTransformerArchitecture::Axk1
            | VulkanTransformerArchitecture::Axk2
            | VulkanTransformerArchitecture::Mistral4
            | VulkanTransformerArchitecture::Youtu => {
                let p = self.peft_decoder_layer_prefix(index);
                let mla = layer
                    .mla_attention
                    .as_ref()
                    .context("DeepSeek-V3 layer is missing MLA attention")?;
                let mut modules = Vec::new();
                if let Some(q_proj) = mla.q_proj.as_ref() {
                    modules.push((format!("{p}.self_attn.q_proj"), q_proj));
                } else {
                    modules.push((
                        format!("{p}.self_attn.q_a_proj"),
                        mla.q_a_proj
                            .as_ref()
                            .context("DeepSeek-V3 MLA is missing q_a_proj")?,
                    ));
                    modules.push((
                        format!("{p}.self_attn.q_b_proj"),
                        mla.q_b_proj
                            .as_ref()
                            .context("DeepSeek-V3 MLA is missing q_b_proj")?,
                    ));
                }
                modules.push((format!("{p}.self_attn.kv_a_proj_with_mqa"), &mla.kv_a_proj));
                modules.push((format!("{p}.self_attn.kv_b_proj"), &mla.kv_b_proj));
                modules.push((format!("{p}.self_attn.o_proj"), &layer.c_proj));
                if let Some(moe) = layer.moe.as_ref() {
                    if let Some(shared) = moe.shared_expert.as_ref() {
                        modules.push((
                            format!("{p}.mlp.shared_experts.gate_proj"),
                            shared
                                .gate
                                .as_ref()
                                .context("DeepSeek-V3 shared experts are missing gate_proj")?,
                        ));
                        modules.push((format!("{p}.mlp.shared_experts.up_proj"), &shared.up));
                        modules
                            .push((format!("{p}.mlp.shared_experts.down_proj"), &shared.down));
                    }
                } else {
                    modules.push((
                        format!("{p}.mlp.gate_proj"),
                        layer
                            .c_gate
                            .as_ref()
                            .context("DeepSeek-V3 dense MLP is missing gate_proj")?,
                    ));
                    modules.push((format!("{p}.mlp.up_proj"), &layer.c_fc));
                    modules.push((format!("{p}.mlp.down_proj"), &layer.c_mlp_proj));
                }
                modules
            }
            VulkanTransformerArchitecture::Mixtral => {
                let p = format!("model.layers.{index}");
                let mut modules = vec![
                    (
                        format!("{p}.self_attn.q_proj"),
                        layer.q_proj.as_ref().context("Mixtral layer is missing q_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.k_proj"),
                        layer.k_proj.as_ref().context("Mixtral layer is missing k_proj")?,
                    ),
                    (
                        format!("{p}.self_attn.v_proj"),
                        layer.v_proj.as_ref().context("Mixtral layer is missing v_proj")?,
                    ),
                    (format!("{p}.self_attn.o_proj"), &layer.c_proj),
                ];
                let sparse = format!("{p}.block_sparse_moe");
                let moe = layer
                    .moe
                    .as_ref()
                    .context("Mixtral layer is missing MoE parameters")?;
                modules.push((format!("{sparse}.gate"), &moe.router));
                modules.push((
                    format!("{sparse}.experts.0.w1"),
                    layer.c_gate.as_ref().context("Mixtral expert 0 is missing w1")?,
                ));
                modules.push((format!("{sparse}.experts.0.w2"), &layer.c_mlp_proj));
                modules.push((format!("{sparse}.experts.0.w3"), &layer.c_fc));
                for (expert_offset, expert) in moe.experts.iter().enumerate() {
                    let expert_index = expert_offset + 1;
                    let expert_prefix = format!("{sparse}.experts.{expert_index}");
                    modules.push((
                        format!("{expert_prefix}.w1"),
                        expert.gate.as_ref().context("Mixtral expert is missing w1")?,
                    ));
                    modules.push((format!("{expert_prefix}.w2"), &expert.down));
                    modules.push((format!("{expert_prefix}.w3"), &expert.up));
                }
                modules
            }
            VulkanTransformerArchitecture::Phimoe
            | VulkanTransformerArchitecture::AriaText
            | VulkanTransformerArchitecture::HyV3
            | VulkanTransformerArchitecture::Qwen2Moe
            | VulkanTransformerArchitecture::Qwen3Moe
            | VulkanTransformerArchitecture::Dots1
            | VulkanTransformerArchitecture::Mellum
            | VulkanTransformerArchitecture::GptOss
            | VulkanTransformerArchitecture::Cohere2Moe
            | VulkanTransformerArchitecture::GraniteMoe
            | VulkanTransformerArchitecture::GraniteMoeShared
            | VulkanTransformerArchitecture::GraniteMoeSwa
            | VulkanTransformerArchitecture::Olmoe
            | VulkanTransformerArchitecture::FlexOlmo
            | VulkanTransformerArchitecture::MiniMaxM2
            | VulkanTransformerArchitecture::MiniMaxM3VLText
            | VulkanTransformerArchitecture::HunYuanMoeV1
            | VulkanTransformerArchitecture::Ernie45Moe
            | VulkanTransformerArchitecture::ExaoneMoe
            | VulkanTransformerArchitecture::Glm4Moe
            | VulkanTransformerArchitecture::SolarOpen => vec![
                (
                    format!("model.layers.{index}.self_attn.q_proj"),
                    layer
                        .q_proj
                        .as_ref()
                        .context("Mixtral layer is missing q_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj"),
                    layer
                        .k_proj
                        .as_ref()
                        .context("Mixtral layer is missing k_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj"),
                    layer
                        .v_proj
                        .as_ref()
                        .context("Mixtral layer is missing v_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj"),
                    &layer.c_proj,
                ),
            ],
            VulkanTransformerArchitecture::StableLm
            | VulkanTransformerArchitecture::Llama
            | VulkanTransformerArchitecture::OpenLlama
            | VulkanTransformerArchitecture::BitNet
            | VulkanTransformerArchitecture::Emu3Text
            | VulkanTransformerArchitecture::Arcee
            | VulkanTransformerArchitecture::Mistral
            | VulkanTransformerArchitecture::Ministral
            | VulkanTransformerArchitecture::Ministral3
            | VulkanTransformerArchitecture::Jais2
            | VulkanTransformerArchitecture::Qwen2
            | VulkanTransformerArchitecture::Qwen3
            | VulkanTransformerArchitecture::Gemma
            | VulkanTransformerArchitecture::Gemma2
            | VulkanTransformerArchitecture::Gemma3
            | VulkanTransformerArchitecture::Gemma4
            | VulkanTransformerArchitecture::VaultGemma
            | VulkanTransformerArchitecture::Granite
            | VulkanTransformerArchitecture::GraniteSwa
            | VulkanTransformerArchitecture::HyperClovax
            | VulkanTransformerArchitecture::Apertus
            | VulkanTransformerArchitecture::Helium
            | VulkanTransformerArchitecture::Cohere
            | VulkanTransformerArchitecture::Cohere2
            | VulkanTransformerArchitecture::CohereCompassText
            | VulkanTransformerArchitecture::SmolLm3
            | VulkanTransformerArchitecture::Olmo
            | VulkanTransformerArchitecture::Olmo2
            | VulkanTransformerArchitecture::Olmo3
            | VulkanTransformerArchitecture::Exaone4
            | VulkanTransformerArchitecture::Nemotron
            | VulkanTransformerArchitecture::SeedOss
            | VulkanTransformerArchitecture::Cwm
            | VulkanTransformerArchitecture::NanoChat
            | VulkanTransformerArchitecture::Ernie45
            | VulkanTransformerArchitecture::HunYuanDenseV1
            | VulkanTransformerArchitecture::EuroBert => {
                let mut modules = vec![
                    (
                        format!("model.layers.{index}.self_attn.q_proj"),
                        layer
                            .q_proj
                            .as_ref()
                            .context("Llama layer is missing q_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.k_proj"),
                        layer
                            .k_proj
                            .as_ref()
                            .context("Llama layer is missing k_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.v_proj"),
                        layer
                            .v_proj
                            .as_ref()
                            .context("Llama layer is missing v_proj")?,
                    ),
                    (
                        format!("model.layers.{index}.self_attn.o_proj"),
                        &layer.c_proj,
                    ),
                ];
                if let Some(gate) = layer.c_gate.as_ref() {
                    modules.push((format!("model.layers.{index}.mlp.gate_proj"), gate));
                }
                if self.config.architecture == VulkanTransformerArchitecture::NanoChat {
                    modules.push((format!("model.layers.{index}.mlp.fc1"), &layer.c_fc));
                    modules.push((format!("model.layers.{index}.mlp.fc2"), &layer.c_mlp_proj));
                } else {
                    modules.push((format!("model.layers.{index}.mlp.up_proj"), &layer.c_fc));
                    modules.push((
                        format!("model.layers.{index}.mlp.down_proj"),
                        &layer.c_mlp_proj,
                    ));
                }
                modules
            }
            VulkanTransformerArchitecture::Phi3 => vec![
                (
                    format!("model.layers.{index}.self_attn.qkv_proj"),
                    layer
                        .c_attn
                        .as_ref()
                        .context("Phi-3 layer is missing qkv_proj")?,
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj"),
                    &layer.c_proj,
                ),
                (
                    format!("model.layers.{index}.mlp.down_proj"),
                    &layer.c_mlp_proj,
                ),
            ],
            VulkanTransformerArchitecture::Albert => {
                unreachable!("ALBERT LoRA is rejected before adapter export")
            }
        };
        Ok(modules)
    }
}
