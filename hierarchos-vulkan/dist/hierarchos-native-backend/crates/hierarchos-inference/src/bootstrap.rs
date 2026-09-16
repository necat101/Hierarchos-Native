use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use rand::{rngs::StdRng, Rng, SeedableRng};
use safetensors::{serialize_to_file, tensor::TensorView, Dtype};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{Error, ModelConfig, Result};

/// Pure-Rust coherent-v9 model bootstrap configuration.
///
/// These defaults mirror the root Python CLI's fresh-training architecture
/// defaults.  The generated package is an ordinary FP32 SafeTensors package,
/// so Vulkan, the pure-Rust inference runtime, and external SafeTensors-aware
/// PyTorch/CUDA consumers all see the same tensor ABI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeBootstrapConfig {
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
    pub rwkv_head_size: usize,
    pub token_adapter_rank: usize,
    pub rosa_max_context: usize,
    pub memory_gate_warmup_steps: usize,
    pub memory_gate_warmup_floor: f32,
    pub detach_every_n_steps: Option<usize>,
    pub training_chunk_size: usize,
    pub seed: u64,
}

impl NativeBootstrapConfig {
    pub fn for_vocab(vocab_size: usize) -> Self {
        let context_dim = 448;
        Self {
            vocab_size,
            context_dim,
            persistent_dim: 128,
            ltm_slots: 1024,
            ltm_key_dim: 128,
            ltm_val_dim: 128,
            ltm_topk: 4,
            h_hidden: context_dim,
            l_hidden: context_dim,
            h_stride: 4,
            max_h_steps: 5,
            max_l_steps: 5,
            min_h_steps: 1,
            rwkv_head_size: choose_head_size(context_dim, None),
            token_adapter_rank: context_dim.min(64),
            rosa_max_context: 512,
            memory_gate_warmup_steps: 2000,
            memory_gate_warmup_floor: 0.10,
            detach_every_n_steps: Some(32),
            training_chunk_size: 256,
            seed: 1337,
        }
    }

    pub fn resolve_auto_geometry(&mut self, explicit_head_size: Option<usize>) {
        self.rwkv_head_size = choose_head_size(self.context_dim, explicit_head_size);
        if self.token_adapter_rank == 0 {
            self.token_adapter_rank = self.context_dim.min(64);
        }
        if self.h_hidden == 0 {
            self.h_hidden = self.context_dim;
        }
        if self.l_hidden == 0 {
            self.l_hidden = self.context_dim;
        }
    }

    fn validate(&self) -> Result<()> {
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
            ("rwkv_head_size", self.rwkv_head_size),
            ("token_adapter_rank", self.token_adapter_rank),
            ("rosa_max_context", self.rosa_max_context),
            ("training_chunk_size", self.training_chunk_size),
        ] {
            if value == 0 {
                return Err(Error::Invalid(format!(
                    "native bootstrap {name} must be positive"
                )));
            }
        }
        if self.h_hidden != self.context_dim {
            return Err(Error::Unsupported(format!(
                "native coherent-v9 currently requires h_hidden={} to equal context_dim={}",
                self.h_hidden, self.context_dim
            )));
        }
        if !self.h_hidden.is_multiple_of(self.rwkv_head_size)
            || !self.l_hidden.is_multiple_of(self.rwkv_head_size)
        {
            return Err(Error::Invalid(format!(
                "rwkv_head_size={} must divide h_hidden={} and l_hidden={}",
                self.rwkv_head_size, self.h_hidden, self.l_hidden
            )));
        }
        if self.ltm_topk > self.ltm_slots {
            return Err(Error::Invalid("ltm_topk must not exceed ltm_slots".into()));
        }
        if self.min_h_steps > self.max_h_steps {
            return Err(Error::Invalid(
                "min_h_steps must not exceed max_h_steps".into(),
            ));
        }
        if !self.memory_gate_warmup_floor.is_finite()
            || !(0.0..=0.95).contains(&self.memory_gate_warmup_floor)
        {
            return Err(Error::Invalid(
                "memory_gate_warmup_floor must be finite and in 0..=0.95".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct OwnedTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl OwnedTensor {
    fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let len = shape.iter().copied().product::<usize>();
        Self {
            shape,
            values: vec![0.0; len],
        }
    }

    fn constant(shape: impl Into<Vec<usize>>, value: f32) -> Self {
        let shape = shape.into();
        let len = shape.iter().copied().product::<usize>();
        Self {
            shape,
            values: vec![value; len],
        }
    }
}

fn choose_head_size(width: usize, requested: Option<usize>) -> usize {
    if let Some(requested) = requested.filter(|value| *value > 0 && width % *value == 0) {
        return requested;
    }
    let mut candidates = (1..=width.min(128))
        .filter(|size| width % size == 0)
        .collect::<Vec<_>>();
    let real_heads = candidates
        .iter()
        .copied()
        .filter(|size| *size >= 16)
        .collect::<Vec<_>>();
    if !real_heads.is_empty() {
        candidates = real_heads;
    }
    candidates
        .into_iter()
        .min_by(|a, b| {
            let da = ((*a as f64 / 64.0).ln()).abs();
            let db = ((*b as f64 / 64.0).ln()).abs();
            da.total_cmp(&db).then_with(|| (*a > 64).cmp(&(*b > 64)))
        })
        .unwrap_or(1)
}

fn rwkv_lora_rank(width: usize, scale: f32) -> usize {
    if width < 128 {
        return 8;
    }
    let rounded = ((scale * (width as f32).sqrt()) / 32.0).round() as usize * 32;
    rounded.max(32)
}

fn normal_sample(rng: &mut StdRng) -> f32 {
    let u1 = (1.0f32 - rng.random::<f32>()).max(f32::MIN_POSITIVE);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn normal_tensor(rng: &mut StdRng, shape: Vec<usize>, stddev: f32) -> OwnedTensor {
    let len = shape.iter().copied().product::<usize>();
    let values = (0..len)
        .map(|_| normal_sample(rng) * stddev)
        .collect::<Vec<_>>();
    OwnedTensor { shape, values }
}

fn uniform_tensor(rng: &mut StdRng, shape: Vec<usize>, bound: f32) -> OwnedTensor {
    let len = shape.iter().copied().product::<usize>();
    let values = (0..len)
        .map(|_| rng.random_range(-bound..=bound))
        .collect::<Vec<_>>();
    OwnedTensor { shape, values }
}

fn linear_weight(rng: &mut StdRng, output: usize, input: usize) -> OwnedTensor {
    uniform_tensor(rng, vec![output, input], 1.0 / (input as f32).sqrt())
}

fn linear_bias(rng: &mut StdRng, output: usize, input: usize) -> OwnedTensor {
    uniform_tensor(rng, vec![output], 1.0 / (input as f32).sqrt())
}

/// Fill a matrix with a deterministic orthogonal basis and scale it by `gain`.
/// This follows the mathematical contract of `torch.nn.init.orthogonal_` while
/// deliberately not promising byte-identical RNG/QR output across frameworks.
fn orthogonal_tensor(rng: &mut StdRng, rows: usize, cols: usize, gain: f32) -> OwnedTensor {
    let mut values = (0..rows * cols)
        .map(|_| normal_sample(rng))
        .collect::<Vec<_>>();
    if rows >= cols {
        for col in 0..cols {
            for prior in 0..col {
                let dot = (0..rows)
                    .map(|row| values[row * cols + col] * values[row * cols + prior])
                    .sum::<f32>();
                for row in 0..rows {
                    values[row * cols + col] -= dot * values[row * cols + prior];
                }
            }
            let norm = (0..rows)
                .map(|row| values[row * cols + col].powi(2))
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            for row in 0..rows {
                values[row * cols + col] = values[row * cols + col] * gain / norm;
            }
        }
    } else {
        for row in 0..rows {
            for prior in 0..row {
                let dot = (0..cols)
                    .map(|col| values[row * cols + col] * values[prior * cols + col])
                    .sum::<f32>();
                for col in 0..cols {
                    values[row * cols + col] -= dot * values[prior * cols + col];
                }
            }
            let norm = (0..cols)
                .map(|col| values[row * cols + col].powi(2))
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            for col in 0..cols {
                values[row * cols + col] = values[row * cols + col] * gain / norm;
            }
        }
    }
    OwnedTensor {
        shape: vec![rows, cols],
        values,
    }
}

fn insert_linear(
    tensors: &mut BTreeMap<String, OwnedTensor>,
    rng: &mut StdRng,
    name: &str,
    output: usize,
    input: usize,
    bias: bool,
) {
    tensors.insert(format!("{name}.weight"), linear_weight(rng, output, input));
    if bias {
        tensors.insert(format!("{name}.bias"), linear_bias(rng, output, input));
    }
}

fn insert_adapter(
    tensors: &mut BTreeMap<String, OwnedTensor>,
    rng: &mut StdRng,
    prefix: &str,
    input: usize,
    rank: usize,
    output: usize,
    bias: f32,
) {
    tensors.insert(
        format!("{prefix}.down.weight"),
        linear_weight(rng, rank, input),
    );
    tensors.insert(
        format!("{prefix}.up.weight"),
        OwnedTensor::zeros(vec![output, rank]),
    );
    tensors.insert(
        format!("{prefix}.bias"),
        OwnedTensor::constant(vec![output], bias),
    );
}

fn insert_rwkv_cell(
    tensors: &mut BTreeMap<String, OwnedTensor>,
    rng: &mut StdRng,
    prefix: &str,
    width: usize,
    head_size: usize,
) {
    let heads = width / head_size;
    let d_decay = rwkv_lora_rank(width, 2.5);
    let d_aaa = rwkv_lora_rank(width, 2.5);
    let d_gate = rwkv_lora_rank(width, 5.0);
    let ratio_0_to_1 = 0.0f32;
    let ratio_1_to_almost0 = 1.0f32;
    let mut x_r = Vec::with_capacity(width);
    let mut x_w = Vec::with_capacity(width);
    let mut x_k = Vec::with_capacity(width);
    let mut x_v = Vec::with_capacity(width);
    let mut x_a = Vec::with_capacity(width);
    let mut x_g = Vec::with_capacity(width);
    let mut w0 = Vec::with_capacity(width);
    let mut a0 = Vec::with_capacity(width);
    let mut k_k = Vec::with_capacity(width);
    for n in 0..width {
        let ddd = n as f32 / width as f32;
        x_r.push(1.0 - ddd.powf(0.2 * ratio_1_to_almost0));
        x_w.push(1.0 - ddd.powf(0.9 * ratio_1_to_almost0));
        x_k.push(1.0 - ddd.powf(0.7 * ratio_1_to_almost0));
        x_v.push(1.0 - ddd.powf(0.7 * ratio_1_to_almost0));
        x_a.push(1.0 - ddd.powf(0.9 * ratio_1_to_almost0));
        x_g.push(1.0 - ddd.powf(0.2 * ratio_1_to_almost0));
        let linear = n as f32 / width.saturating_sub(1).max(1) as f32 - 0.5;
        let z = if head_size > 1 {
            ((n % head_size) as f32 - (head_size - 1) as f32 / 2.0) / ((head_size - 1) as f32 / 2.0)
        } else {
            0.0
        };
        let zigzag = z * z.abs();
        let www = -6.0
            + 6.0
                * (n as f32 / width.saturating_sub(1).max(1) as f32)
                    .powf(1.0 + ratio_0_to_1.powf(0.3));
        w0.push(www + 0.5 + zigzag * 2.5);
        a0.push(-0.19 + zigzag * 0.3 + linear * 0.4);
        k_k.push(0.71 - linear * 0.1);
    }
    for (suffix, values) in [
        ("x_r", x_r),
        ("x_w", x_w),
        ("x_k", x_k),
        ("x_v", x_v),
        ("x_a", x_a),
        ("x_g", x_g),
        ("w0", w0),
        ("a0", a0),
        ("k_k", k_k),
    ] {
        tensors.insert(
            format!("{prefix}.{suffix}"),
            OwnedTensor {
                shape: vec![1, width],
                values,
            },
        );
    }
    tensors.insert(
        format!("{prefix}.w1"),
        OwnedTensor::zeros(vec![width, d_decay]),
    );
    tensors.insert(
        format!("{prefix}.w2"),
        orthogonal_tensor(rng, d_decay, width, 0.1),
    );
    tensors.insert(
        format!("{prefix}.a1"),
        OwnedTensor::zeros(vec![width, d_aaa]),
    );
    tensors.insert(
        format!("{prefix}.a2"),
        orthogonal_tensor(rng, d_aaa, width, 0.1),
    );
    tensors.insert(
        format!("{prefix}.g1"),
        OwnedTensor::zeros(vec![width, d_gate]),
    );
    tensors.insert(
        format!("{prefix}.g2"),
        orthogonal_tensor(rng, d_gate, width, 0.1),
    );
    tensors.insert(
        format!("{prefix}.k_a"),
        OwnedTensor::constant(vec![1, width], 1.02),
    );
    tensors.insert(
        format!("{prefix}.r_k"),
        OwnedTensor::constant(vec![heads, head_size], -0.04),
    );
    tensors.insert(
        format!("{prefix}.x_k_cm"),
        OwnedTensor::zeros(vec![1, width]),
    );
    for norm in ["ln1", "ln2", "ln_x"] {
        tensors.insert(
            format!("{prefix}.{norm}.weight"),
            OwnedTensor::constant(vec![width], 1.0),
        );
        tensors.insert(
            format!("{prefix}.{norm}.bias"),
            OwnedTensor::zeros(vec![width]),
        );
    }
    let half_bound = 0.5 / (width as f32).sqrt();
    tensors.insert(
        format!("{prefix}.receptance.weight"),
        uniform_tensor(rng, vec![width, width], half_bound),
    );
    tensors.insert(
        format!("{prefix}.key.weight"),
        uniform_tensor(rng, vec![width, width], 0.05 / (width as f32).sqrt()),
    );
    tensors.insert(
        format!("{prefix}.value.weight"),
        uniform_tensor(rng, vec![width, width], half_bound),
    );
    let tiny_bound = 0.01 / (width as f32).sqrt();
    tensors.insert(
        format!("{prefix}.output.weight"),
        uniform_tensor(rng, vec![width, width], tiny_bound),
    );
    tensors.insert(
        format!("{prefix}.value_cm.weight"),
        uniform_tensor(rng, vec![width, width * 4], tiny_bound),
    );
    tensors.insert(
        format!("{prefix}.key_cm.weight"),
        orthogonal_tensor(rng, width * 4, width, 2.0),
    );
}

fn architecture_contract(config: &NativeBootstrapConfig) -> Value {
    let commitment_threshold = 0.1f32 / config.context_dim as f32;
    json!({
        "architecture_contract_schema_version": 3,
        "architecture_revision": "coherent-v9",
        "vocab_size": config.vocab_size,
        "context_dim": config.context_dim,
        "persistent_dim": config.persistent_dim,
        "ltm_slots": config.ltm_slots,
        "ltm_key_dim": config.ltm_key_dim,
        "ltm_val_dim": config.ltm_val_dim,
        "ltm_topk": config.ltm_topk,
        "h_hidden": config.h_hidden,
        "l_hidden": config.l_hidden,
        "rwkv_head_size": config.rwkv_head_size,
        "h_rwkv_head_size": config.rwkv_head_size,
        "l_rwkv_head_size": config.rwkv_head_size,
        "rwkv_n_layer_hint": 2,
        "h_stride": config.h_stride,
        "max_h_steps": config.max_h_steps,
        "max_l_steps": config.max_l_steps,
        "core_recurrence_version": 2,
        "drift_recurrence_mode": "state-derived",
        "rwkv_state_readout_mode": "explicit-output",
        "manager_state_commit_mode": "hard-selected",
        "manager_compute_mode": "hard-masked",
        "min_h_steps": config.min_h_steps,
        "h_halt_thresh": 0.9,
        "act_depth_temperature": 0.05,
        "l_conv_atol": 0.0001,
        "commitment_cost_mode": "mean-square",
        "commitment_threshold": commitment_threshold,
        "drift_delta_scale": 1.0,
        "detach_every_n_steps": config.detach_every_n_steps,
        "full_sample_bptt": false,
        "inference_logit_parity": true,
        "inference_recurrence_mode": "tbptt",
        "halt_logit_clamp": 30.0,
        "recurrent_state_clamp": 50.0,
        "context_state_clamp": 50.0,
        "drift_state_clamp": 5.0,
        "drift_norm_clamp": 0.0,
        "activation_clamp": 100.0,
        "rwkv_channel_mix_key_clamp": 12.0,
        "rwkv_channel_mix_deepembed_clamp": 4.0,
        "inference_logit_clamp": 0.0,
        "use_deepembed": true,
        "deepembed_mode": "shared-factorized",
        "use_rosa": true,
        "rosa_embedding_mode": "shared-factorized",
        "token_adapter_rank": config.token_adapter_rank,
        "memory_token_routers": true,
        "rosa_max_context": config.rosa_max_context,
        "enforce_rosa_max_context": true,
        "rosa_zero_no_prediction": true,
        "isolate_batch_ltm": true,
        "ltm_training_mode": "read-only",
        "ltm_lr": 0.001,
        "ltm_momentum": 0.9,
        "ltm_weight_decay": 0.0001,
        "ltm_forget_rate": 0.01,
        "ltm_score_grad_scale": 1.0,
        "ltm_time_feature_mode": "metadata-only",
        "reference_chunk_len": config.training_chunk_size,
        "training_chunk_size": config.training_chunk_size,
        "allow_untrained_hebbian_writer": false,
        "memory_gate_warmup_steps": config.memory_gate_warmup_steps,
        "memory_gate_warmup_floor": config.memory_gate_warmup_floor,
        "adaptive_ponder": true,
        "ponder_objective": "symmetric-huber",
        "ponder_target_scale": 0.5,
        "ponder_huber_beta": 0.5,
        "ponder_loss_weight": 0.01,
        "encourage_thinking": false,
        "commitment_loss_weight": 0.5,
        "max_commitment_cost_for_backward": 2.0,
        "max_ponder_cost_for_backward": 0.0,
        "z_loss_weight": 0.0001,
        "ltm_value_alignment_weight": 0.01,
        "ltm_value_alignment_stride": 8,
        "ltm_value_alignment_min_updates": 100,
        "ltm_value_alignment_ready_threshold": 0.95,
        "ltm_value_alignment_ema_decay": 0.95,
        "ltm_value_writer_max_norm": 64.0
    })
}

fn contract_sha256(contract: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(contract)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn package_config(config: &NativeBootstrapConfig, contract: Value, digest: &str) -> Value {
    let mut value = contract.clone();
    let object = value
        .as_object_mut()
        .expect("architecture contract is an object");
    object.remove("architecture_contract_schema_version");
    object.insert("format_version".into(), json!(1));
    object.insert("model_type".into(), json!("hierarchos"));
    object.insert("compile".into(), json!(false));
    object.insert("gradient_checkpointing".into(), json!(false));
    object.insert("architecture_contract".into(), contract);
    object.insert("architecture_contract_sha256".into(), json!(digest));
    object.insert("val_proj_alignment_updates".into(), json!(0));
    object.insert("val_proj_alignment_last".into(), Value::Null);
    object.insert("val_proj_alignment_ema".into(), Value::Null);
    object.insert("val_proj_alignment_best".into(), Value::Null);
    object.insert("val_proj_writer_norm".into(), Value::Null);
    object.insert("val_proj_trained".into(), json!(false));
    object.insert("bootstrap_seed".into(), json!(config.seed));
    value
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Create a complete coherent-v9 package without importing or launching Python.
pub fn initialize_model_package(
    output_dir: impl AsRef<Path>,
    config: &NativeBootstrapConfig,
) -> Result<ModelConfig> {
    config.validate()?;
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let model_path = output_dir.join("model.safetensors");
    if model_path.exists() {
        return Err(Error::Invalid(format!(
            "refusing to overwrite existing bootstrap model {}",
            model_path.display()
        )));
    }

    let contract = architecture_contract(config);
    let digest = contract_sha256(&contract)?;
    let full_config = package_config(config, contract, &digest);
    let pretty = serde_json::to_vec_pretty(&full_config)?;
    fs::write(output_dir.join("hierarchos_config.json"), &pretty)?;
    fs::write(output_dir.join("hierarchos_rust_config.json"), &pretty)?;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut tensors = BTreeMap::<String, OwnedTensor>::new();
    let c = config.context_dim;
    let h = config.h_hidden;
    let l = config.l_hidden;
    let rank = config.token_adapter_rank;

    insert_adapter(
        &mut tensors,
        &mut rng,
        "h_deepembed_adapter",
        c,
        rank,
        h * 4,
        1.0,
    );
    insert_adapter(
        &mut tensors,
        &mut rng,
        "l_deepembed_adapter",
        c,
        rank,
        l * 4,
        1.0,
    );
    insert_adapter(&mut tensors, &mut rng, "rosa_adapter", c, rank, c, 0.0);
    tensors.insert(
        "rosa_gate_logit".into(),
        OwnedTensor::constant(vec![], -1.0),
    );
    tensors.insert("ltm_gate_logit".into(), OwnedTensor::constant(vec![], -2.0));
    tensors.insert("memory_gate_warmup_step".into(), OwnedTensor::zeros(vec![]));
    insert_linear(&mut tensors, &mut rng, "rosa_router", 1, c, true);
    tensors.insert("rosa_router.weight".into(), OwnedTensor::zeros(vec![1, c]));
    tensors.insert("rosa_router.bias".into(), OwnedTensor::zeros(vec![1]));
    insert_linear(&mut tensors, &mut rng, "ltm_router", 1, c, true);
    tensors.insert("ltm_router.weight".into(), OwnedTensor::zeros(vec![1, c]));
    tensors.insert("ltm_router.bias".into(), OwnedTensor::zeros(vec![1]));
    tensors.insert(
        "persistent".into(),
        normal_tensor(&mut rng, vec![config.persistent_dim], 0.02),
    );
    tensors.insert(
        "ltm.keys".into(),
        normal_tensor(&mut rng, vec![config.ltm_slots, config.ltm_key_dim], 0.02),
    );
    let mut ltm_vals = orthogonal_tensor(&mut rng, config.ltm_slots, config.ltm_val_dim, 1.0);
    for value in &mut ltm_vals.values {
        *value *= 0.02;
    }
    tensors.insert("ltm.vals".into(), ltm_vals);
    tensors.insert(
        "ltm.fast_vals".into(),
        OwnedTensor::zeros(vec![config.ltm_slots, config.ltm_val_dim]),
    );
    tensors.insert(
        "ltm._mom_vals".into(),
        OwnedTensor::zeros(vec![config.ltm_slots, config.ltm_val_dim]),
    );
    tensors.insert(
        "ltm.timestamps".into(),
        OwnedTensor::zeros(vec![config.ltm_slots]),
    );
    insert_linear(
        &mut tensors,
        &mut rng,
        "qproj",
        config.ltm_key_dim,
        c * 2,
        false,
    );
    insert_linear(
        &mut tensors,
        &mut rng,
        "val_proj",
        config.ltm_val_dim,
        c,
        false,
    );
    insert_linear(
        &mut tensors,
        &mut rng,
        "in_proj",
        c,
        c + config.persistent_dim + config.ltm_val_dim * config.ltm_topk,
        true,
    );
    tensors.insert(
        "l_feedback_proj.weight".into(),
        normal_tensor(&mut rng, vec![h, l], 0.01),
    );
    insert_rwkv_cell(&mut tensors, &mut rng, "h_rnn", h, config.rwkv_head_size);
    insert_rwkv_cell(&mut tensors, &mut rng, "l_rnn", l, config.rwkv_head_size);
    insert_linear(&mut tensors, &mut rng, "h_to_context", c, h, true);
    insert_linear(&mut tensors, &mut rng, "h_halt_proj", 1, h, true);
    tensors.insert(
        "h_halt_proj.bias".into(),
        OwnedTensor::constant(vec![1], -(config.max_h_steps.max(2) as f32 - 1.0).ln()),
    );
    insert_linear(&mut tensors, &mut rng, "l_input_proj", l, c * 2, true);
    tensors.insert(
        "context_drift_proj.weight".into(),
        normal_tensor(&mut rng, vec![c, l], 0.01),
    );
    insert_linear(&mut tensors, &mut rng, "l_to_out", c, l, true);
    tensors.insert(
        "out_norm.weight".into(),
        OwnedTensor::constant(vec![c], 1.0),
    );
    tensors.insert("out_norm.bias".into(), OwnedTensor::zeros(vec![c]));
    tensors.insert(
        "lm_head.weight".into(),
        linear_weight(&mut rng, config.vocab_size, c),
    );
    let half_dim = config.ltm_val_dim / 2;
    let time_freqs = if half_dim == 0 {
        vec![]
    } else if half_dim == 1 {
        vec![1.0]
    } else {
        let scale = 10000.0f32.ln() / (half_dim - 1) as f32;
        (0..half_dim)
            .map(|index| (-(index as f32) * scale).exp())
            .collect()
    };
    tensors.insert(
        "time_freqs".into(),
        OwnedTensor {
            shape: vec![half_dim],
            values: time_freqs,
        },
    );

    let owned = tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                tensor.shape.clone(),
                f32_bytes(&tensor.values),
            )
        })
        .collect::<Vec<_>>();
    let mut views = Vec::with_capacity(owned.len());
    for (name, shape, bytes) in &owned {
        views.push((
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes)?,
        ));
    }
    let metadata = HashMap::from([
        ("format".to_string(), "hierarchos-rust-fp32-v1".to_string()),
        (
            "architecture_revision".to_string(),
            "coherent-v9".to_string(),
        ),
        ("architecture_contract_sha256".to_string(), digest),
        (
            "bootstrap_backend".to_string(),
            "rust-native-v1".to_string(),
        ),
    ]);
    serialize_to_file(views, Some(metadata), &model_path)?;
    ModelConfig::from_model_dir(output_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HierarchosModel;

    #[test]
    fn native_bootstrap_package_loads_in_pure_rust_runtime() {
        let mut config = NativeBootstrapConfig::for_vocab(64);
        config.context_dim = 32;
        config.h_hidden = 32;
        config.l_hidden = 32;
        config.persistent_dim = 8;
        config.ltm_slots = 16;
        config.ltm_key_dim = 8;
        config.ltm_val_dim = 8;
        config.ltm_topk = 2;
        config.h_stride = 2;
        config.max_h_steps = 3;
        config.max_l_steps = 2;
        config.rwkv_head_size = 32;
        config.token_adapter_rank = 32;
        config.rosa_max_context = 8;
        config.memory_gate_warmup_steps = 10;
        config.memory_gate_warmup_floor = 0.35;
        config.detach_every_n_steps = None;
        let dir = std::env::temp_dir().join(format!(
            "hierarchos-native-bootstrap-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        initialize_model_package(&dir, &config).unwrap();
        let model = HierarchosModel::load(&dir).unwrap();
        assert_eq!(model.config().vocab_size, 64);
        assert_eq!(model.config().context_dim, 32);
        fs::remove_dir_all(&dir).unwrap();
    }
}
