use std::{fs, path::Path, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    math::{
        finite_clamp, finite_clamp_vec, gelu, l2_norm_clamp, layer_norm, sigmoid, silu, Linear,
        Matrix,
    },
    rosa::{RosaState, RosaStateSnapshot},
    rwkv::{RwkvCell, RwkvState},
    sampler::Sampler,
    weights::WeightLoader,
};

pub const RUNTIME_STATE_INTERCHANGE_KIND: &str = "hierarchos_runtime_state_interchange";
pub const RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION: u32 = 1;
pub const RWKV_V8_MATRIX_PACKED_LAYOUT: &str = "rwkv_v8_matrix_packed";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuntimeRwkvStateSnapshot {
    pub layout: String,
    pub state_readout_mode: String,
    pub hidden: usize,
    pub head_size: usize,
    pub state_size: usize,
    /// Canonical tensor shape `[1, hidden, state_size]`.
    pub shape: [usize; 3],
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuntimeStateSnapshot {
    pub kind: String,
    pub schema_version: u32,
    pub architecture_revision: String,
    pub architecture_contract_sha256: Option<String>,
    pub position: usize,
    pub history: Vec<u32>,
    pub h_state: RuntimeRwkvStateSnapshot,
    pub l_state: RuntimeRwkvStateSnapshot,
    pub prev_context: Vec<f32>,
    pub target_context: Vec<f32>,
    pub final_drift: Vec<f32>,
    pub fast_vals: Vec<f32>,
    pub rosa: RosaStateSnapshot,
}

fn default_true() -> bool {
    true
}

fn default_state_clamp() -> f32 {
    50.0
}
fn default_context_clamp() -> f32 {
    50.0
}
fn default_drift_clamp() -> f32 {
    5.0
}
fn default_activation_clamp() -> f32 {
    100.0
}
fn default_key_clamp() -> f32 {
    12.0
}
fn default_deepembed_clamp() -> f32 {
    4.0
}
fn default_halt_logit_clamp() -> f32 {
    30.0
}
fn default_drift_delta_scale() -> f32 {
    1.0
}
fn default_commitment_cost_mode() -> String {
    "mean-square".to_string()
}
fn default_commitment_threshold() -> f32 {
    0.1 / 448.0
}
fn default_act_depth_temperature() -> f32 {
    0.05
}
fn default_ltm_score_grad_scale() -> f32 {
    1.0
}
fn default_ltm_value_alignment_stride() -> usize {
    1
}
fn default_ltm_value_alignment_min_updates() -> u64 {
    100
}
fn default_ltm_value_alignment_ready_threshold() -> f32 {
    0.95
}
fn default_ltm_value_alignment_ema_decay() -> f32 {
    0.95
}
fn default_ltm_value_writer_max_norm() -> f32 {
    64.0
}
fn deserialize_null_default_u64<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<u64>::deserialize(deserializer)?.unwrap_or_default())
}
fn deserialize_null_default_bool<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub format_version: u32,
    pub architecture_revision: String,
    /// Cross-backend learned-function identity emitted by the Python exporter.
    /// Older packages may omit it, but when present the native runtime carries
    /// and validates the canonical 256-bit hexadecimal identifier.
    #[serde(default)]
    pub architecture_contract_sha256: Option<String>,
    pub vocab_size: usize,
    pub context_dim: usize,
    pub persistent_dim: usize,
    pub ltm_slots: usize,
    pub ltm_key_dim: usize,
    pub ltm_val_dim: usize,
    pub ltm_topk: usize,
    pub h_hidden: usize,
    pub l_hidden: usize,
    pub h_stride: usize,
    pub max_h_steps: usize,
    pub max_l_steps: usize,
    pub min_h_steps: usize,
    pub h_halt_thresh: f32,
    pub l_conv_atol: f32,
    #[serde(default = "default_commitment_cost_mode")]
    pub commitment_cost_mode: String,
    #[serde(default = "default_commitment_threshold")]
    pub commitment_threshold: f32,
    #[serde(default = "default_act_depth_temperature")]
    pub act_depth_temperature: f32,
    pub h_rwkv_head_size: usize,
    pub l_rwkv_head_size: usize,
    pub token_adapter_rank: usize,
    pub rosa_max_context: usize,

    #[serde(default = "default_true")]
    pub use_deepembed: bool,
    #[serde(default = "default_true")]
    pub use_rosa: bool,
    #[serde(default = "default_true")]
    pub memory_token_routers: bool,
    #[serde(default = "default_true")]
    pub enforce_rosa_max_context: bool,
    #[serde(default = "default_true")]
    pub inference_logit_parity: bool,

    pub deepembed_mode: String,
    pub rosa_embedding_mode: String,
    pub rwkv_state_readout_mode: String,
    pub manager_compute_mode: String,
    pub manager_state_commit_mode: String,
    pub ltm_time_feature_mode: String,
    #[serde(default = "default_ltm_score_grad_scale")]
    pub ltm_score_grad_scale: f32,
    /// Training-only coherent-v9 writer-controller policy. These fields live in
    /// the shared model contract so a Vulkan trainer and a PyTorch/CUDA trainer
    /// make the same stride/readiness decisions from the same package.
    #[serde(default)]
    pub ltm_value_alignment_weight: f32,
    #[serde(default = "default_ltm_value_alignment_stride")]
    pub ltm_value_alignment_stride: usize,
    #[serde(default = "default_ltm_value_alignment_min_updates")]
    pub ltm_value_alignment_min_updates: u64,
    #[serde(default = "default_ltm_value_alignment_ready_threshold")]
    pub ltm_value_alignment_ready_threshold: f32,
    #[serde(default = "default_ltm_value_alignment_ema_decay")]
    pub ltm_value_alignment_ema_decay: f32,
    #[serde(default = "default_ltm_value_writer_max_norm")]
    pub ltm_value_writer_max_norm: f32,
    #[serde(default, deserialize_with = "deserialize_null_default_u64")]
    pub val_proj_alignment_updates: u64,
    #[serde(default)]
    pub val_proj_alignment_last: Option<f32>,
    #[serde(default)]
    pub val_proj_alignment_ema: Option<f32>,
    #[serde(default)]
    pub val_proj_alignment_best: Option<f32>,
    #[serde(default)]
    pub val_proj_writer_norm: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_null_default_bool")]
    pub val_proj_trained: bool,

    #[serde(default = "default_state_clamp")]
    pub recurrent_state_clamp: f32,
    #[serde(default = "default_context_clamp")]
    pub context_state_clamp: f32,
    #[serde(default = "default_drift_clamp")]
    pub drift_state_clamp: f32,
    #[serde(default)]
    pub drift_norm_clamp: f32,
    #[serde(default = "default_activation_clamp")]
    pub activation_clamp: f32,
    #[serde(default = "default_halt_logit_clamp")]
    pub halt_logit_clamp: f32,
    #[serde(default = "default_key_clamp")]
    pub rwkv_channel_mix_key_clamp: f32,
    #[serde(default = "default_deepembed_clamp")]
    pub rwkv_channel_mix_deepembed_clamp: f32,
    #[serde(default = "default_drift_delta_scale")]
    pub drift_delta_scale: f32,
    #[serde(default)]
    pub inference_logit_clamp: f32,
    #[serde(default)]
    pub memory_gate_warmup_steps: f32,
    #[serde(default)]
    pub memory_gate_warmup_floor: f32,
}

impl ModelConfig {
    /// Load and validate the learned-function contract from an exported model
    /// package without materializing the model weights.
    ///
    /// This is intentionally public so alternate execution backends (for
    /// example the Vulkan trainer) can consume exactly the same architecture
    /// contract as the native inference runtime.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config_path = model_dir.as_ref().join("hierarchos_rust_config.json");
        let config: Self = serde_json::from_slice(&fs::read(config_path)?)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate this config against the coherent-v9 native runtime contract.
    pub fn validate_runtime_contract(&self) -> Result<()> {
        self.validate()
    }

    fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return Err(Error::Unsupported(format!(
                "export format version {} (runtime supports 1)",
                self.format_version
            )));
        }
        if let Some(hash) = self.architecture_contract_sha256.as_deref() {
            if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(Error::Invalid(format!(
                    "architecture_contract_sha256 must be a 64-character hexadecimal SHA-256 digest; got {hash:?}"
                )));
            }
        }
        let expected = [
            (
                self.architecture_revision.as_str(),
                "coherent-v9",
                "architecture_revision",
            ),
            (
                self.deepembed_mode.as_str(),
                "shared-factorized",
                "deepembed_mode",
            ),
            (
                self.rosa_embedding_mode.as_str(),
                "shared-factorized",
                "rosa_embedding_mode",
            ),
            (
                self.rwkv_state_readout_mode.as_str(),
                "explicit-output",
                "rwkv_state_readout_mode",
            ),
            (
                self.manager_compute_mode.as_str(),
                "hard-masked",
                "manager_compute_mode",
            ),
            (
                self.manager_state_commit_mode.as_str(),
                "hard-selected",
                "manager_state_commit_mode",
            ),
            (
                self.commitment_cost_mode.as_str(),
                "mean-square",
                "commitment_cost_mode",
            ),
            (
                self.ltm_time_feature_mode.as_str(),
                "metadata-only",
                "ltm_time_feature_mode",
            ),
        ];
        for (actual, wanted, name) in expected {
            if actual != wanted {
                return Err(Error::Unsupported(format!(
                    "{name}={actual:?}; expected {wanted:?}"
                )));
            }
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("context_dim", self.context_dim),
            ("ltm_slots", self.ltm_slots),
            ("ltm_key_dim", self.ltm_key_dim),
            ("ltm_val_dim", self.ltm_val_dim),
            ("ltm_topk", self.ltm_topk),
            ("h_hidden", self.h_hidden),
            ("l_hidden", self.l_hidden),
            ("h_stride", self.h_stride),
            ("max_h_steps", self.max_h_steps),
            ("max_l_steps", self.max_l_steps),
            ("min_h_steps", self.min_h_steps),
            ("token_adapter_rank", self.token_adapter_rank),
        ] {
            if value == 0 {
                return Err(Error::Invalid(format!("{name} must be positive")));
            }
        }
        if !self.ltm_value_alignment_weight.is_finite() || self.ltm_value_alignment_weight < 0.0 {
            return Err(Error::Unsupported(format!(
                "ltm_value_alignment_weight={} must be finite and non-negative",
                self.ltm_value_alignment_weight
            )));
        }
        if self.ltm_value_alignment_stride == 0 || self.ltm_value_alignment_min_updates == 0 {
            return Err(Error::Unsupported(
                "ltm_value_alignment_stride and ltm_value_alignment_min_updates must be positive"
                    .to_string(),
            ));
        }
        if !self.ltm_value_alignment_ready_threshold.is_finite()
            || self.ltm_value_alignment_ready_threshold < 0.0
            || !self.ltm_value_alignment_ema_decay.is_finite()
            || !(0.0..1.0).contains(&self.ltm_value_alignment_ema_decay)
            || !self.ltm_value_writer_max_norm.is_finite()
            || self.ltm_value_writer_max_norm < 0.0
        {
            return Err(Error::Unsupported(
                "invalid LTM value-alignment readiness controller parameters".to_string(),
            ));
        }
        if self.use_rosa && self.enforce_rosa_max_context && self.rosa_max_context == 0 {
            return Err(Error::Invalid(
                "rosa_max_context must be positive when bounded ROSA is enabled".into(),
            ));
        }
        if self.h_hidden != self.context_dim {
            return Err(Error::Unsupported(format!(
                "h_hidden={} must equal context_dim={}",
                self.h_hidden, self.context_dim
            )));
        }
        if self.h_rwkv_head_size == 0
            || self.l_rwkv_head_size == 0
            || !self.h_hidden.is_multiple_of(self.h_rwkv_head_size)
            || !self.l_hidden.is_multiple_of(self.l_rwkv_head_size)
        {
            return Err(Error::Invalid(
                "RWKV head sizes must divide their cell widths".into(),
            ));
        }
        if self.h_stride == 0 || self.max_h_steps == 0 || self.max_l_steps == 0 {
            return Err(Error::Invalid(
                "h_stride, max_h_steps, and max_l_steps must be positive".into(),
            ));
        }
        if self.min_h_steps == 0 || self.min_h_steps > self.max_h_steps {
            return Err(Error::Invalid(
                "min_h_steps must be in the range 1..=max_h_steps".into(),
            ));
        }
        if !self.h_halt_thresh.is_finite() || !(0.0..=1.0).contains(&self.h_halt_thresh) {
            return Err(Error::Invalid(
                "h_halt_thresh must be finite and in the range 0..=1".into(),
            ));
        }
        if !self.commitment_threshold.is_finite() || self.commitment_threshold < 0.0 {
            return Err(Error::Invalid(
                "commitment_threshold must be finite and non-negative".into(),
            ));
        }
        if !self.act_depth_temperature.is_finite() || self.act_depth_temperature <= 0.0 {
            return Err(Error::Invalid(
                "act_depth_temperature must be finite and positive".into(),
            ));
        }
        for (name, value) in [
            ("recurrent_state_clamp", self.recurrent_state_clamp),
            ("context_state_clamp", self.context_state_clamp),
            ("drift_state_clamp", self.drift_state_clamp),
            ("activation_clamp", self.activation_clamp),
            ("halt_logit_clamp", self.halt_logit_clamp),
            ("l_conv_atol", self.l_conv_atol),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::Invalid(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        for (name, value) in [
            ("drift_norm_clamp", self.drift_norm_clamp),
            ("drift_delta_scale", self.drift_delta_scale),
            (
                "rwkv_channel_mix_key_clamp",
                self.rwkv_channel_mix_key_clamp,
            ),
            (
                "rwkv_channel_mix_deepembed_clamp",
                self.rwkv_channel_mix_deepembed_clamp,
            ),
            ("memory_gate_warmup_steps", self.memory_gate_warmup_steps),
            ("memory_gate_warmup_floor", self.memory_gate_warmup_floor),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::Invalid(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if self.memory_gate_warmup_floor > 0.95 {
            return Err(Error::Invalid(
                "memory_gate_warmup_floor must be in the range 0..=0.95".into(),
            ));
        }
        if self.ltm_topk == 0 || self.ltm_slots == 0 || self.ltm_topk > self.ltm_slots {
            return Err(Error::Invalid(
                "ltm_topk must be in the range 1..=ltm_slots".into(),
            ));
        }
        if !self.ltm_score_grad_scale.is_finite() || self.ltm_score_grad_scale < 0.0 {
            return Err(Error::Invalid(
                "ltm_score_grad_scale must be finite and nonnegative".into(),
            ));
        }
        if !self.use_deepembed || !self.use_rosa || !self.memory_token_routers {
            return Err(Error::Unsupported(
                "phase-1 coherent-v9 runtime currently requires use_deepembed=true, use_rosa=true, and memory_token_routers=true".into(),
            ));
        }
        if !self.inference_logit_parity {
            return Err(Error::Unsupported(
                "coherent-v9 requires inference_logit_parity=true".into(),
            ));
        }
        if self.inference_logit_clamp != 0.0 {
            return Err(Error::Unsupported(
                "phase-1 Rust coherent-v9 runtime requires inference_logit_clamp=0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct SharedAdapter {
    down: Linear,
    up: Linear,
    bias: Arc<[f32]>,
}

impl SharedAdapter {
    fn load(
        loader: &WeightLoader,
        prefix: &str,
        input: usize,
        rank: usize,
        output: usize,
    ) -> Result<Self> {
        Ok(Self {
            down: loader.linear(&format!("{prefix}.down"), rank, input, false)?,
            up: loader.linear(&format!("{prefix}.up"), output, rank, false)?,
            bias: loader.vector(&format!("{prefix}.bias"), output)?,
        })
    }

    fn forward_normalized(&self, normalized: &[f32]) -> Vec<f32> {
        let mut hidden = self.down.forward(normalized);
        for value in &mut hidden {
            *value = silu(*value);
        }
        let mut out = self.up.forward(&hidden);
        for (value, &bias) in out.iter_mut().zip(self.bias.iter()) {
            *value += bias;
        }
        out
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let normalized = layer_norm(x, None, None, 1e-5);
        self.forward_normalized(&normalized)
    }
}

#[derive(Clone)]
pub struct RuntimeState {
    h_state: RwkvState,
    l_state: RwkvState,
    prev_context: Vec<f32>,
    target_context: Vec<f32>,
    final_drift: Vec<f32>,
    fast_vals: Vec<f32>,
    rosa: RosaState,
    position: usize,
    history: Vec<u32>,
}

impl RuntimeState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }
}

pub struct HierarchosModel {
    config: ModelConfig,
    lm_head: Matrix,
    h_deepembed: SharedAdapter,
    l_deepembed: SharedAdapter,
    rosa_adapter: SharedAdapter,
    rosa_gate_logit: f32,
    rosa_router: Linear,
    ltm_gate_logit: f32,
    ltm_router: Linear,
    memory_gate_warmup_step: f32,
    persistent: Arc<[f32]>,
    ltm_keys: Matrix,
    ltm_vals: Matrix,
    initial_fast_vals: Arc<[f32]>,
    qproj: Linear,
    in_proj: Linear,
    l_feedback_proj: Linear,
    h_rnn: RwkvCell,
    h_to_context: Linear,
    h_halt_proj: Linear,
    l_input_proj: Linear,
    l_rnn: RwkvCell,
    context_drift_proj: Linear,
    l_to_out: Linear,
    out_norm_weight: Arc<[f32]>,
    out_norm_bias: Arc<[f32]>,
}

impl HierarchosModel {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let weights_path = model_dir.join("model.safetensors");
        let config = ModelConfig::from_model_dir(model_dir)?;
        let loader = WeightLoader::open(&weights_path)?;

        let c = config.context_dim;
        let l = config.l_hidden;
        let rank = config.token_adapter_rank;
        let lm_head = loader.matrix("lm_head.weight", config.vocab_size, c)?;
        let h_deepembed =
            SharedAdapter::load(&loader, "h_deepembed_adapter", c, rank, config.h_hidden * 4)?;
        let l_deepembed = SharedAdapter::load(&loader, "l_deepembed_adapter", c, rank, l * 4)?;
        let rosa_adapter = SharedAdapter::load(&loader, "rosa_adapter", c, rank, c)?;

        Ok(Self {
            h_rnn: RwkvCell::load(
                &loader,
                "h_rnn",
                config.h_hidden,
                config.h_rwkv_head_size,
                config.recurrent_state_clamp,
                config.rwkv_channel_mix_key_clamp,
                config.rwkv_channel_mix_deepembed_clamp,
            )?,
            l_rnn: RwkvCell::load(
                &loader,
                "l_rnn",
                l,
                config.l_rwkv_head_size,
                config.recurrent_state_clamp,
                config.rwkv_channel_mix_key_clamp,
                config.rwkv_channel_mix_deepembed_clamp,
            )?,
            lm_head,
            h_deepembed,
            l_deepembed,
            rosa_adapter,
            rosa_gate_logit: loader.scalar("rosa_gate_logit")?,
            rosa_router: loader.linear("rosa_router", 1, c, true)?,
            ltm_gate_logit: loader.scalar("ltm_gate_logit")?,
            ltm_router: loader.linear("ltm_router", 1, c, true)?,
            memory_gate_warmup_step: loader.scalar("memory_gate_warmup_step")?,
            persistent: loader.vector("persistent", config.persistent_dim)?,
            ltm_keys: loader.matrix("ltm.keys", config.ltm_slots, config.ltm_key_dim)?,
            ltm_vals: loader.matrix("ltm.vals", config.ltm_slots, config.ltm_val_dim)?,
            initial_fast_vals: loader
                .flat("ltm.fast_vals", config.ltm_slots * config.ltm_val_dim)?,
            qproj: loader.linear("qproj", config.ltm_key_dim, c * 2, false)?,
            in_proj: loader.linear(
                "in_proj",
                c,
                c + config.persistent_dim + config.ltm_topk * config.ltm_val_dim,
                true,
            )?,
            l_feedback_proj: loader.linear("l_feedback_proj", config.h_hidden, l, false)?,
            h_to_context: loader.linear("h_to_context", c, config.h_hidden, true)?,
            h_halt_proj: loader.linear("h_halt_proj", 1, config.h_hidden, true)?,
            l_input_proj: loader.linear("l_input_proj", l, c * 2, true)?,
            context_drift_proj: loader.linear("context_drift_proj", c, l, false)?,
            l_to_out: loader.linear("l_to_out", c, l, true)?,
            out_norm_weight: loader.vector("out_norm.weight", c)?,
            out_norm_bias: loader.vector("out_norm.bias", c)?,
            config,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn new_state(&self) -> RuntimeState {
        RuntimeState {
            h_state: self.h_rnn.zero_state(),
            l_state: self.l_rnn.zero_state(),
            prev_context: vec![0.0; self.config.context_dim],
            target_context: vec![0.0; self.config.context_dim],
            final_drift: vec![0.0; self.config.context_dim],
            fast_vals: self.initial_fast_vals.to_vec(),
            rosa: RosaState::new(),
            position: 0,
            history: Vec::new(),
        }
    }

    fn snapshot_rwkv_state(
        &self,
        state: &RwkvState,
        hidden: usize,
        head_size: usize,
    ) -> Result<RuntimeRwkvStateSnapshot> {
        let state_size = RwkvState::EXPLICIT_OUTPUT_MATRIX_OFFSET + head_size;
        Ok(RuntimeRwkvStateSnapshot {
            layout: RWKV_V8_MATRIX_PACKED_LAYOUT.to_owned(),
            state_readout_mode: "explicit-output".to_owned(),
            hidden,
            head_size,
            state_size,
            shape: [1, hidden, state_size],
            values: state.to_explicit_output_packed(hidden, head_size)?,
        })
    }

    fn restore_rwkv_state(
        &self,
        label: &str,
        snapshot: &RuntimeRwkvStateSnapshot,
        hidden: usize,
        head_size: usize,
    ) -> Result<RwkvState> {
        let state_size = RwkvState::EXPLICIT_OUTPUT_MATRIX_OFFSET + head_size;
        if snapshot.layout != RWKV_V8_MATRIX_PACKED_LAYOUT
            || snapshot.state_readout_mode != "explicit-output"
            || snapshot.hidden != hidden
            || snapshot.head_size != head_size
            || snapshot.state_size != state_size
            || snapshot.shape != [1, hidden, state_size]
        {
            return Err(Error::Invalid(format!(
                "{label} runtime-state geometry/layout does not match this model: saved layout={} readout={} hidden={} head_size={} state_size={} shape={:?}; expected layout={RWKV_V8_MATRIX_PACKED_LAYOUT} readout=explicit-output hidden={hidden} head_size={head_size} state_size={state_size} shape={:?}",
                snapshot.layout,
                snapshot.state_readout_mode,
                snapshot.hidden,
                snapshot.head_size,
                snapshot.state_size,
                snapshot.shape,
                [1, hidden, state_size]
            )));
        }
        RwkvState::from_explicit_output_packed(hidden, head_size, &snapshot.values)
    }

    /// Materialize a backend-neutral inference-state snapshot.
    ///
    /// The recurrent tensors deliberately use the same packed coherent-v9
    /// layout as PyTorch and Vulkan (`[1, C, 4 + head_size]`), making this JSON
    /// an actual handoff format rather than a serialization of Rust internals.
    pub fn snapshot_runtime_state(&self, state: &RuntimeState) -> Result<RuntimeStateSnapshot> {
        if state.position != state.history.len() {
            return Err(Error::Invalid(format!(
                "runtime-state position/history mismatch: position={} history={}",
                state.position,
                state.history.len()
            )));
        }
        if state
            .history
            .iter()
            .any(|token| *token as usize >= self.config.vocab_size)
        {
            return Err(Error::Invalid(
                "runtime-state history contains a token outside the model vocabulary".into(),
            ));
        }
        self.validate_runtime_state_vectors(
            &state.prev_context,
            &state.target_context,
            &state.final_drift,
            &state.fast_vals,
        )?;

        Ok(RuntimeStateSnapshot {
            kind: RUNTIME_STATE_INTERCHANGE_KIND.to_owned(),
            schema_version: RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION,
            architecture_revision: self.config.architecture_revision.clone(),
            architecture_contract_sha256: self.config.architecture_contract_sha256.clone(),
            position: state.position,
            history: state.history.clone(),
            h_state: self.snapshot_rwkv_state(
                &state.h_state,
                self.config.h_hidden,
                self.config.h_rwkv_head_size,
            )?,
            l_state: self.snapshot_rwkv_state(
                &state.l_state,
                self.config.l_hidden,
                self.config.l_rwkv_head_size,
            )?,
            prev_context: state.prev_context.clone(),
            target_context: state.target_context.clone(),
            final_drift: state.final_drift.clone(),
            fast_vals: state.fast_vals.clone(),
            rosa: state.rosa.snapshot(),
        })
    }

    fn validate_runtime_state_vectors(
        &self,
        prev_context: &[f32],
        target_context: &[f32],
        final_drift: &[f32],
        fast_vals: &[f32],
    ) -> Result<()> {
        let expected_fast_vals = self
            .config
            .ltm_slots
            .checked_mul(self.config.ltm_val_dim)
            .ok_or_else(|| Error::Invalid("runtime-state LTM geometry overflow".into()))?;
        for (name, values, expected) in [
            ("prev_context", prev_context, self.config.context_dim),
            ("target_context", target_context, self.config.context_dim),
            ("final_drift", final_drift, self.config.context_dim),
            ("fast_vals", fast_vals, expected_fast_vals),
        ] {
            if values.len() != expected {
                return Err(Error::Invalid(format!(
                    "runtime-state {name} has {} values; expected {expected}",
                    values.len()
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid(format!(
                    "runtime-state {name} contains non-finite values"
                )));
            }
        }
        Ok(())
    }

    /// Restore a portable inference snapshot after strict learned-function,
    /// recurrent-geometry, tensor-shape, vocabulary, and ROSA validation.
    pub fn restore_runtime_state(&self, snapshot: &RuntimeStateSnapshot) -> Result<RuntimeState> {
        if snapshot.kind != RUNTIME_STATE_INTERCHANGE_KIND
            || snapshot.schema_version != RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION
        {
            return Err(Error::Unsupported(format!(
                "runtime-state interchange kind/schema {:?}/{} is unsupported",
                snapshot.kind, snapshot.schema_version
            )));
        }
        if snapshot.architecture_revision != self.config.architecture_revision
            || snapshot.architecture_contract_sha256 != self.config.architecture_contract_sha256
        {
            return Err(Error::Invalid(format!(
                "runtime-state learned-function identity does not match this model: saved revision={:?} contract={:?}; model revision={:?} contract={:?}",
                snapshot.architecture_revision,
                snapshot.architecture_contract_sha256,
                self.config.architecture_revision,
                self.config.architecture_contract_sha256
            )));
        }
        if snapshot.position != snapshot.history.len() {
            return Err(Error::Invalid(format!(
                "runtime-state position/history mismatch: position={} history={}",
                snapshot.position,
                snapshot.history.len()
            )));
        }
        if snapshot
            .history
            .iter()
            .any(|token| *token as usize >= self.config.vocab_size)
            || snapshot
                .rosa
                .tokens
                .iter()
                .any(|token| *token as usize >= self.config.vocab_size)
        {
            return Err(Error::Invalid(
                "runtime-state contains a token outside the model vocabulary".into(),
            ));
        }
        if snapshot.rosa.tokens.len() > snapshot.history.len()
            || !snapshot.history.ends_with(&snapshot.rosa.tokens)
        {
            return Err(Error::Invalid(
                "runtime-state ROSA history is not a suffix of the canonical token history".into(),
            ));
        }
        self.validate_runtime_state_vectors(
            &snapshot.prev_context,
            &snapshot.target_context,
            &snapshot.final_drift,
            &snapshot.fast_vals,
        )?;

        Ok(RuntimeState {
            h_state: self.restore_rwkv_state(
                "h_state",
                &snapshot.h_state,
                self.config.h_hidden,
                self.config.h_rwkv_head_size,
            )?,
            l_state: self.restore_rwkv_state(
                "l_state",
                &snapshot.l_state,
                self.config.l_hidden,
                self.config.l_rwkv_head_size,
            )?,
            prev_context: snapshot.prev_context.clone(),
            target_context: snapshot.target_context.clone(),
            final_drift: snapshot.final_drift.clone(),
            fast_vals: snapshot.fast_vals.clone(),
            rosa: RosaState::from_snapshot(&snapshot.rosa)?,
            position: snapshot.position,
            history: snapshot.history.clone(),
        })
    }

    pub fn save_runtime_state_json(
        &self,
        state: &RuntimeState,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let snapshot = self.snapshot_runtime_state(state)?;
        let bytes = serde_json::to_vec_pretty(&snapshot)?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load_runtime_state_json(&self, path: impl AsRef<Path>) -> Result<RuntimeState> {
        let bytes = fs::read(path)?;
        let snapshot: RuntimeStateSnapshot = serde_json::from_slice(&bytes)?;
        self.restore_runtime_state(&snapshot)
    }

    #[inline]
    fn embedding(&self, token: u32) -> Result<&[f32]> {
        if token as usize >= self.config.vocab_size {
            return Err(Error::Invalid(format!(
                "token id {token} is outside vocabulary {}",
                self.config.vocab_size
            )));
        }
        Ok(self.lm_head.row(token as usize))
    }

    /// Execute the checkpoint-stable token projection seam shared with the
    /// Vulkan trainer, using an already-gated LTM payload.
    ///
    /// This intentionally stops short of ROSA prediction, qproj, and top-k LTM
    /// retrieval. Those are stateful parts of `step`; exposing this narrower
    /// seam gives alternate training backends a production-native parity oracle
    /// without duplicating Hierarchos' `persistent + in_proj + GELU` logic.
    pub fn project_token_frontend(
        &self,
        token: u32,
        token_residual: Option<&[f32]>,
        gated_ltm_values: &[f32],
    ) -> Result<Vec<f32>> {
        let mut token_x = self.embedding(token)?.to_vec();
        if let Some(residual) = token_residual {
            if residual.len() != self.config.context_dim {
                return Err(Error::Invalid(format!(
                    "token_residual has {} values; expected {}",
                    residual.len(),
                    self.config.context_dim
                )));
            }
            if residual.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid(
                    "token_residual contains non-finite values".into(),
                ));
            }
            for (value, delta) in token_x.iter_mut().zip(residual.iter()) {
                *value += *delta;
            }
        }
        self.project_gated_frontend(&token_x, gated_ltm_values)
    }

    fn project_gated_frontend(
        &self,
        token_x: &[f32],
        gated_ltm_values: &[f32],
    ) -> Result<Vec<f32>> {
        if token_x.len() != self.config.context_dim {
            return Err(Error::Invalid(format!(
                "token feature width {} does not match context_dim {}",
                token_x.len(),
                self.config.context_dim
            )));
        }
        let expected_ltm = self.config.ltm_topk * self.config.ltm_val_dim;
        if gated_ltm_values.len() != expected_ltm {
            return Err(Error::Invalid(format!(
                "gated LTM payload has {} values; expected {}",
                gated_ltm_values.len(),
                expected_ltm
            )));
        }
        if token_x
            .iter()
            .chain(gated_ltm_values.iter())
            .any(|value| !value.is_finite())
        {
            return Err(Error::Invalid(
                "token front-end input contains non-finite values".into(),
            ));
        }

        let mut mac_in = Vec::with_capacity(
            self.config.context_dim
                + self.config.persistent_dim
                + self.config.ltm_topk * self.config.ltm_val_dim,
        );
        mac_in.extend_from_slice(token_x);
        mac_in.extend_from_slice(&self.persistent);
        mac_in.extend_from_slice(gated_ltm_values);
        let mut enc = self.in_proj.forward(&mac_in);
        for value in &mut enc {
            *value = gelu(*value);
        }
        finite_clamp_vec(&mut enc, 30.0);
        Ok(enc)
    }

    fn memory_gate_floor(&self) -> f32 {
        if self.config.memory_gate_warmup_steps <= 0.0
            || self.config.memory_gate_warmup_floor <= 0.0
        {
            return 0.0;
        }
        let floor = self.config.memory_gate_warmup_floor.clamp(0.0, 0.95);
        let progress =
            (self.memory_gate_warmup_step / self.config.memory_gate_warmup_steps).clamp(0.0, 1.0);
        floor * (1.0 - progress)
    }

    fn apply_gate_floor(&self, gate: f32) -> f32 {
        let floor = self.memory_gate_floor();
        floor + (1.0 - floor) * gate
    }

    fn retrieve_ltm(&self, q: &[f32], fast_vals: &[f32]) -> Vec<f32> {
        let scale = (self.config.ltm_key_dim as f32).powf(-0.5);
        let mut scores: Vec<(usize, f32)> = (0..self.config.ltm_slots)
            .map(|slot| {
                let score = crate::math::dot(q, self.ltm_keys.row(slot)) * scale;
                (
                    slot,
                    if score.is_nan() {
                        f32::NEG_INFINITY
                    } else {
                        score
                    },
                )
            })
            .collect();
        let topk = self.config.ltm_topk.min(scores.len());
        if topk < scores.len() {
            scores.select_nth_unstable_by(topk, |a, b| b.1.total_cmp(&a.1));
            scores.truncate(topk);
        }
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut out = Vec::with_capacity(self.config.ltm_topk * self.config.ltm_val_dim);
        for (slot, _) in scores {
            let slow = self.ltm_vals.row(slot);
            let fast_start = slot * self.config.ltm_val_dim;
            let fast = &fast_vals[fast_start..fast_start + self.config.ltm_val_dim];
            for i in 0..self.config.ltm_val_dim {
                out.push(slow[i] + fast[i]);
            }
        }
        while out.len() < self.config.ltm_topk * self.config.ltm_val_dim {
            out.push(0.0);
        }
        out
    }

    fn worker_step(
        &self,
        enc: &[f32],
        static_context: &[f32],
        real_state: &RwkvState,
        initial_drift: &[f32],
        deepembed: &[f32],
    ) -> (Vec<f32>, RwkvState, Vec<f32>) {
        let mut current_drift = initial_drift.to_vec();
        finite_clamp_vec(&mut current_drift, self.config.drift_state_clamp);
        l2_norm_clamp(&mut current_drift, self.config.drift_norm_clamp);
        let current_enc = enc;

        let mut dynamic_context: Vec<f32> = static_context
            .iter()
            .zip(current_drift.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let mut l_input_vec = Vec::with_capacity(self.config.context_dim * 2);
        l_input_vec.extend_from_slice(current_enc);
        l_input_vec.extend_from_slice(&dynamic_context);
        let mut l_input = self.l_input_proj.forward(&l_input_vec);
        finite_clamp_vec(&mut l_input, self.config.recurrent_state_clamp);

        let mut shadow = real_state.clone();
        for _ in 0..self.config.max_l_steps {
            let (mut l_out, candidate_shadow) = self.l_rnn.step(&l_input, &shadow, Some(deepembed));
            finite_clamp_vec(&mut l_out, self.config.activation_clamp);
            let mut drift_delta = self.context_drift_proj.forward(&l_out);
            for value in &mut drift_delta {
                *value = value.tanh() * self.config.drift_delta_scale;
            }
            for i in 0..current_drift.len() {
                current_drift[i] += drift_delta[i];
            }
            finite_clamp_vec(&mut current_drift, self.config.drift_state_clamp);
            l2_norm_clamp(&mut current_drift, self.config.drift_norm_clamp);
            shadow = candidate_shadow;

            dynamic_context.clear();
            dynamic_context.extend(
                static_context
                    .iter()
                    .zip(current_drift.iter())
                    .map(|(&a, &b)| a + b),
            );
            l_input_vec.clear();
            l_input_vec.extend_from_slice(current_enc);
            l_input_vec.extend_from_slice(&dynamic_context);
            l_input = self.l_input_proj.forward(&l_input_vec);
            finite_clamp_vec(&mut l_input, self.config.recurrent_state_clamp);

            let mean_abs =
                drift_delta.iter().map(|v| v.abs()).sum::<f32>() / drift_delta.len() as f32;
            if mean_abs < self.config.l_conv_atol {
                break;
            }
        }

        // Commit exactly one real worker transition after shadow refinement.
        let (mut final_l_out, next_l_state) =
            self.l_rnn.step(&l_input, real_state, Some(deepembed));
        finite_clamp_vec(&mut final_l_out, self.config.activation_clamp);
        let projected = self.l_to_out.forward(&final_l_out);
        let mut final_enc: Vec<f32> = current_enc
            .iter()
            .zip(projected.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        finite_clamp_vec(&mut final_enc, self.config.activation_clamp);
        (final_enc, next_l_state, current_drift)
    }

    /// Advance one token and return logits predicting the following token.
    pub fn step(&self, token: u32, state: &mut RuntimeState) -> Result<Vec<f32>> {
        let raw_embedding = self.embedding(token)?.to_vec();
        let normalized_embedding = layer_norm(&raw_embedding, None, None, 1e-5);
        let h_deepembed = self.h_deepembed.forward_normalized(&normalized_embedding);
        let l_deepembed = self.l_deepembed.forward_normalized(&normalized_embedding);

        let mut token_x = raw_embedding.clone();
        if self.config.use_rosa {
            let cap = if self.config.enforce_rosa_max_context {
                self.config.rosa_max_context
            } else {
                0
            };
            let prediction = state.rosa.predict_and_push(token, cap);
            if let Some(predicted) = prediction.filter(|&p| (p as usize) < self.config.vocab_size) {
                let rosa_features = self.embedding(predicted)?;
                let rosa_emb = self.rosa_adapter.forward(rosa_features);
                let router = if self.config.memory_token_routers {
                    self.rosa_router.forward(&raw_embedding)[0]
                } else {
                    0.0
                };
                let gate = self
                    .apply_gate_floor(sigmoid(finite_clamp(self.rosa_gate_logit + router, 50.0)));
                for i in 0..token_x.len() {
                    token_x[i] += gate * rosa_emb[i];
                }
            }
        }

        let mut q_input = Vec::with_capacity(self.config.context_dim * 2);
        q_input.extend_from_slice(&token_x);
        q_input.extend_from_slice(&state.prev_context);
        let mut q = self.qproj.forward(&q_input);
        finite_clamp_vec(&mut q, 12.0);
        let mut ltm_values = self.retrieve_ltm(&q, &state.fast_vals);
        let ltm_router = if self.config.memory_token_routers {
            self.ltm_router.forward(&token_x)[0]
        } else {
            0.0
        };
        let ltm_gate = self.apply_gate_floor(sigmoid(finite_clamp(
            self.ltm_gate_logit + ltm_router,
            50.0,
        )));
        for value in &mut ltm_values {
            *value *= ltm_gate;
        }

        let enc = self.project_gated_frontend(&token_x, &ltm_values)?;

        let l_feedback = self.l_feedback_proj.forward(&state.l_state.output);
        let mut enc_with_feedback: Vec<f32> = enc
            .iter()
            .zip(l_feedback.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        finite_clamp_vec(&mut enc_with_feedback, self.config.activation_clamp);

        let (mut h_out_real, real_h_state) =
            self.h_rnn
                .step(&enc_with_feedback, &state.h_state, Some(&h_deepembed));
        finite_clamp_vec(&mut h_out_real, self.config.activation_clamp);
        state.h_state = real_h_state;

        let abs_t = state.position;
        if abs_t.is_multiple_of(self.config.h_stride) {
            state.prev_context.clone_from(&state.target_context);

            let mut outputs = vec![h_out_real.clone()];
            let mut states = vec![state.h_state.clone()];
            let mut halt_probs = vec![sigmoid(finite_clamp(
                self.h_halt_proj.forward(&h_out_real)[0],
                self.config.halt_logit_clamp,
            ))
            .clamp(1e-6, 1.0 - 1e-6)];
            let mut survival = 1.0 - halt_probs[0];
            let mut halted =
                self.config.min_h_steps <= 1 && (1.0 - survival) >= self.config.h_halt_thresh;
            let mut shadow = state.h_state.clone();

            for step_idx in 0..self.config.max_h_steps.saturating_sub(1) {
                if halted {
                    break;
                }
                let (mut ponder, next_shadow) =
                    self.h_rnn
                        .step(&enc_with_feedback, &shadow, Some(&h_deepembed));
                finite_clamp_vec(&mut ponder, self.config.activation_clamp);
                shadow = next_shadow;
                let p = sigmoid(finite_clamp(
                    self.h_halt_proj.forward(&ponder)[0],
                    self.config.halt_logit_clamp,
                ))
                .clamp(1e-6, 1.0 - 1e-6);
                outputs.push(ponder);
                states.push(shadow.clone());
                halt_probs.push(p);
                survival *= 1.0 - p;
                let completed = step_idx + 2;
                if completed >= self.config.min_h_steps
                    && (1.0 - survival) >= self.config.h_halt_thresh
                {
                    halted = true;
                }
            }

            // hard_act_selection: first cumulative CDF crossing after min_h_steps,
            // falling back to the final computed candidate.
            let mut cumulative_survival = 1.0f32;
            let mut selected = outputs.len() - 1;
            for (idx, &p) in halt_probs.iter().enumerate() {
                cumulative_survival *= 1.0 - p;
                if idx + 1 >= self.config.min_h_steps
                    && (1.0 - cumulative_survival) >= self.config.h_halt_thresh
                {
                    selected = idx;
                    break;
                }
            }
            state.h_state = states[selected].clone();
            let mut final_h = outputs[selected].clone();
            finite_clamp_vec(&mut final_h, self.config.activation_clamp);
            state.target_context = self.h_to_context.forward(&final_h);
            finite_clamp_vec(&mut state.target_context, self.config.context_state_clamp);
        }

        let step_in_stride = abs_t % self.config.h_stride;
        let alpha = step_in_stride as f32 / self.config.h_stride as f32;
        let mut sliding_context: Vec<f32> = state
            .prev_context
            .iter()
            .zip(state.target_context.iter())
            .map(|(&previous, &target)| previous + alpha * (target - previous))
            .collect();
        finite_clamp_vec(&mut sliding_context, self.config.context_state_clamp);

        // coherent-v9 state-derived drift recurrence.
        let mut initial_drift = self.context_drift_proj.forward(&state.l_state.output);
        for value in &mut initial_drift {
            *value = value.tanh();
        }
        finite_clamp_vec(&mut initial_drift, self.config.drift_state_clamp);
        l2_norm_clamp(&mut initial_drift, self.config.drift_norm_clamp);

        let (final_enc, next_l_state, mut final_drift) = self.worker_step(
            &enc,
            &sliding_context,
            &state.l_state,
            &initial_drift,
            &l_deepembed,
        );
        state.l_state = next_l_state;
        finite_clamp_vec(&mut final_drift, self.config.drift_state_clamp);
        l2_norm_clamp(&mut final_drift, self.config.drift_norm_clamp);
        state.final_drift = final_drift;

        let mut final_norm = layer_norm(
            &final_enc,
            Some(&self.out_norm_weight),
            Some(&self.out_norm_bias),
            1e-5,
        );
        // PyTorch applies the model-wide activation safety boundary after
        // out_norm and before the tied LM projection. With the production
        // default (100) this is usually inactive, which allowed the native
        // runtime to drift unnoticed until a deliberately saturated safety
        // qualification lowered the ceiling. Keep this finite-preserving: an
        // existing NaN/Inf must still reach the fail-closed logit guard.
        finite_clamp_vec(&mut final_norm, self.config.activation_clamp);
        let logits = self.lm_head.matvec(&final_norm);
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(
                "non-finite language-model logits; refusing to sanitize coherent-v9 output".into(),
            ));
        }

        state.position += 1;
        state.history.push(token);
        Ok(logits)
    }

    pub fn prefill(&self, tokens: &[u32], state: &mut RuntimeState) -> Result<Vec<Vec<f32>>> {
        tokens
            .iter()
            .map(|&token| self.step(token, state))
            .collect()
    }

    /// Advance a prompt while retaining only the final logit vector.
    ///
    /// Autoregressive generation only needs the logits from the final prompt
    /// token. Keeping every intermediate vocabulary-sized vector can add
    /// hundreds of megabytes of transient memory for long prompts, so native
    /// frontends should prefer this helper over [`Self::prefill`].
    pub fn prefill_last(&self, tokens: &[u32], state: &mut RuntimeState) -> Result<Vec<f32>> {
        let mut last = None;
        for &token in tokens {
            last = Some(self.step(token, state)?);
        }
        last.ok_or_else(|| Error::Invalid("prefill prompt must contain at least one token".into()))
    }

    pub fn generate_ids(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::Invalid(
                "generation prompt must contain at least one token".into(),
            ));
        }
        let mut state = self.new_state();
        let mut logits = self.prefill_last(prompt, &mut state)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let token = sampler.sample(&logits, state.history());
            generated.push(token);
            logits = self.step(token, &mut state)?;
        }
        Ok(generated)
    }
}
