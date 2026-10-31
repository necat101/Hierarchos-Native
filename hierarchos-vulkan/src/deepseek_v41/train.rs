//! Native optimizer state for V4.1 LoRA adaptation.
//!
//! The V4.1 adapter representation is a set of host-resident FP32 factor pairs
//! (`a: [rank, in]`, `b: [out, rank]`) that the graph uploads per projection, so
//! adaptation is trained with host AdamW over those factors rather than a
//! device-resident optimizer slot per packed weight. This module owns the
//! mutable masters, the moment estimates, the step count and the checkpoint
//! contract; `terminal` owns the loss-side adjoints and `linear_grad` produces
//! FP32 projection/factor gradients. `model_train` connects the complete graph.
//! Absent gradients skip the corresponding parameter entirely;
//! factor-local counters drive bias correction and checkpoint resume.
//!
//! AdamW is gated against `torch.optim.AdamW`'s single-tensor CPU path. Native
//! FP32 staging applies decay before moment updates, evaluates bias corrections
//! in f64 and rounds their scalars to f32, then applies
//! `param + ((-step_size) * m) / denom`. Checkpoint metadata preserves the
//! previous FP32 policy when resuming older optimizer files.
//! `validation/validate_deepseek_v41_adapter_training.py` pins PyTorch's own
//! parameters and moments after each step.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::peft::DeepseekV41LoraAdapter;

#[path = "train_conditional.rs"]
mod conditional;
pub use conditional::DeepseekV41AdapterParameterGradient;

/// Hyperparameters of the adapter optimizer. Defaults match `torch.optim.AdamW`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeepseekV41AdamWConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for DeepseekV41AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl DeepseekV41AdamWConfig {
    fn validate(&self) -> Result<()> {
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            bail!("V4.1 adapter learning rate must be finite and positive");
        }
        for (label, value) in [
            ("beta1", self.beta1),
            ("beta2", self.beta2),
            ("weight_decay", self.weight_decay),
        ] {
            if !(value.is_finite() && (0.0..1.0).contains(&value)) {
                bail!("V4.1 adapter {label} must be in [0, 1)");
            }
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            bail!("V4.1 adapter epsilon must be finite and positive");
        }
        Ok(())
    }
}

/// Raw FP32 bits of one factor's AdamW state, in checkpoint layout.
///
/// `step == 0` means the factor has no optimizer slot yet, which is how a
/// lazily allocated state is represented in a checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct DeepseekV41AdapterFactorState<'a> {
    pub step: u64,
    pub avg: &'a [u32],
    pub square: &'a [u32],
}

/// `d loss` for one adapted projection, in the adapter's own factor layout.
#[derive(Debug, Clone)]
pub struct DeepseekV41AdapterGradient {
    /// Canonical checkpoint tensor name, as reported by `targets()`.
    pub target: String,
    /// `[rank, in_features]`, FP32, row-major.
    pub grad_a: Vec<f32>,
    /// `[out_features, rank]`, FP32, row-major.
    pub grad_b: Vec<f32>,
}

#[derive(Debug, Clone)]
struct TrainableFactors {
    rank: usize,
    alpha: f64,
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AdamWSlot {
    step_a: u64,
    step_b: u64,
    exp_avg_a: Vec<f32>,
    exp_avg_sq_a: Vec<f32>,
    exp_avg_b: Vec<f32>,
    exp_avg_sq_b: Vec<f32>,
}

#[derive(Debug,Clone,Copy,Default,PartialEq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
enum AdamWMathPolicy {
    #[default]
    LegacyF32V1,
    CpuSingleTensorV2,
}

/// Mutable adapter masters plus AdamW moments for the whole adapter.
#[derive(Debug, Clone)]
pub struct DeepseekV41AdapterTrainer {
    math_policy: AdamWMathPolicy,
    /// Source adapter: its config, name and checksum are what the trained
    /// factors are exported under.
    template: DeepseekV41LoraAdapter,
    source_sha256: String,
    config: DeepseekV41AdamWConfig,
    step: u64,
    factors: BTreeMap<String, TrainableFactors>,
    slots: BTreeMap<String, AdamWSlot>,
}

const OPTIMIZER_STATE: &str = "optimizer_state.safetensors";
const OPTIMIZER_METADATA: &str = "optimizer_state.json";
const OPTIMIZER_SCHEMA: u32 = 4;

#[derive(Serialize, Deserialize)]
struct OptimizerMetadata {
    #[serde(default)]
    math_policy: AdamWMathPolicy,
    schema: u32,
    step: u64,
    config: DeepseekV41AdamWConfig,
    source_sha256: String,
    targets: Vec<String>,
    #[serde(default)]
    adapter_snapshot_sha256: String,
    #[serde(default)]
    adapter_file_sha256: String,
    #[serde(default)]
    config_file_sha256: String,
    #[serde(default)]
    optimizer_file_sha256: String,
}

impl DeepseekV41AdapterTrainer {
    /// Copy an adapter's factors into trainable masters, zeroing the moments.
    pub fn new(adapter: &DeepseekV41LoraAdapter, config: DeepseekV41AdamWConfig) -> Result<Self> {
        config.validate()?;
        if adapter.is_empty() {
            bail!("V4.1 adapter training requires at least one adapted projection");
        }
        let mut factors = BTreeMap::new();
        let mut slots = BTreeMap::new();
        for (target, _, _) in adapter.targets() {
            let module = adapter
                .module_for_tensor(&format!("{target}.weight"))
                .with_context(|| format!("V4.1 adapter module {target} is missing"))?;
            let factors_entry = TrainableFactors {
                rank: module.rank,
                alpha: module.alpha,
                a: module.a.to_vec(),
                b: module.b.to_vec(),
            };
            if factors_entry
                .a
                .iter()
                .chain(&factors_entry.b)
                .any(|v| !v.is_finite())
            {
                bail!("V4.1 adapter factors for {target} must be finite");
            }
            let slot = AdamWSlot {
                step_a: 0,
                step_b: 0,
                exp_avg_a: Vec::new(),
                exp_avg_sq_a: Vec::new(),
                exp_avg_b: Vec::new(),
                exp_avg_sq_b: Vec::new(),
            };
            factors.insert(target.to_owned(), factors_entry);
            slots.insert(target.to_owned(), slot);
        }
        Ok(Self {
            math_policy:AdamWMathPolicy::CpuSingleTensorV2,
            template: adapter.clone(),
            source_sha256: adapter.source_sha256.clone(),
            config,
            step: 0,
            factors,
            slots,
        })
    }

    /// Optimizer calls, including calls with no active parameter gradients.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn config(&self) -> DeepseekV41AdamWConfig {
        self.config
    }

    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.factors.keys().map(String::as_str)
    }

    /// Overwrite one target's factors and optimizer state from raw FP32 bits.
    ///
    /// A trajectory gate that compares several consecutive training steps pins
    /// each step's input state with this, so the forward, backward and update
    /// arithmetic under test starts from exactly the values the reference used.
    /// Without it a first-step rounding difference propagates through the
    /// optimizer and can flip a near-tied discrete decision many steps later,
    /// which measures the trajectory's conditioning rather than the kernels.
    pub fn load_state_bits(
        &mut self,
        target: &str,
        a_bits: &[u32],
        b_bits: &[u32],
        state_a: Option<DeepseekV41AdapterFactorState<'_>>,
        state_b: Option<DeepseekV41AdapterFactorState<'_>>,
    ) -> Result<()> {
        let factors = self.factors.get_mut(target).with_context(|| {
            format!("V4.1 adapter state target {target} is not part of this adapter")
        })?;
        if a_bits.len() != factors.a.len() || b_bits.len() != factors.b.len() {
            bail!("V4.1 adapter state for {target} has the wrong factor geometry");
        }
        let (a_len, b_len) = (factors.a.len(), factors.b.len());
        write_bits(&mut factors.a, a_bits, target, "A")?;
        write_bits(&mut factors.b, b_bits, target, "B")?;
        let slot = self
            .slots
            .get_mut(target)
            .with_context(|| format!("V4.1 optimizer slot for {target} is missing"))?;
        for (state, len, step, avg, square, label) in [
            (
                state_a,
                a_len,
                &mut slot.step_a,
                &mut slot.exp_avg_a,
                &mut slot.exp_avg_sq_a,
                "A",
            ),
            (
                state_b,
                b_len,
                &mut slot.step_b,
                &mut slot.exp_avg_b,
                &mut slot.exp_avg_sq_b,
                "B",
            ),
        ] {
            let Some(state) = state else {
                *step = 0;
                avg.clear();
                square.clear();
                continue;
            };
            if state.avg.len() != len || state.square.len() != len {
                bail!("V4.1 adapter state for {target} {label} has the wrong moment geometry");
            }
            *step = state.step;
            avg.resize(state.avg.len(), 0.0);
            square.resize(state.square.len(), 0.0);
            write_bits(avg, state.avg, target, label)?;
            write_bits(square, state.square, target, label)?;
        }
        Ok(())
    }

    /// The trained factors as a normal adapter, ready for `export`.
    pub fn adapter(&self) -> Result<DeepseekV41LoraAdapter> {
        self.template.with_factors(|target| {
            self.factors
                .get(target)
                .map(|factors| (factors.a.clone(), factors.b.clone()))
        })
    }

    /// Write optimizer moments, step count and hyperparameters beside an
    /// adapter. The adapter files themselves are not written here: call
    /// `adapter().export(dir)` first, then this, so a resumed run reads the
    /// same factor values it saved.
    pub fn save(&self, directory: impl AsRef<Path>) -> Result<()> {
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
        let directory = directory.as_ref();
        if directory.join(OPTIMIZER_STATE).exists() || directory.join(OPTIMIZER_METADATA).exists() {
            bail!("V4.1 optimizer checkpoint already exists; save to a fresh adapter directory");
        }
        let exported = DeepseekV41LoraAdapter::load(directory)?;
        let snapshot = self.adapter()?.snapshot_digest()?;
        if exported.snapshot_digest()? != snapshot {
            bail!("V4.1 optimizer checkpoint adapter does not contain the current trained factors/config");
        }
        let adapter_file_sha256 =
            hash_file(&directory.join(super::peft::DEEPSEEK_V41_LORA_WEIGHTS))?;
        let config_file_sha256 = hash_file(&directory.join(super::peft::DEEPSEEK_V41_LORA_CONFIG))?;
        let mut tensors: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> = BTreeMap::new();
        let mut push = |name: String, values: &[f32], shape: Vec<usize>| -> Result<()> {
            if values.iter().any(|v| !v.is_finite()) {
                bail!("V4.1 optimizer checkpoint contains non-finite moments at {name}");
            }
            tensors.insert(
                name,
                (
                    Dtype::F32,
                    shape,
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
            );
            Ok(())
        };
        for (target, slot) in &self.slots {
            let factors = &self.factors[target];
            validate_slot(target, slot, factors, self.step)?;
            let width = factors.a.len() / factors.rank.max(1);
            let out = factors.b.len() / factors.rank.max(1);
            for (label, values, shape) in [
                ("exp_avg_a", &slot.exp_avg_a, vec![factors.rank, width]),
                (
                    "exp_avg_sq_a",
                    &slot.exp_avg_sq_a,
                    vec![factors.rank, width],
                ),
                ("exp_avg_b", &slot.exp_avg_b, vec![out, factors.rank]),
                ("exp_avg_sq_b", &slot.exp_avg_sq_b, vec![out, factors.rank]),
            ] {
                if values.is_empty() { continue; }
                push(format!("{target}.{label}"), values, shape)?;
            }
        }
        drop(push);
        // Counter payloads share the optimizer-file integrity binding with
        // the moments; they are not editable, unbound JSON metadata.
        for (target, slot) in &self.slots {
            for (name, value) in [("step_a", slot.step_a), ("step_b", slot.step_b)] {
                tensors.insert(
                    format!("{target}.{name}"),
                    (Dtype::U64, vec![1], value.to_le_bytes().to_vec()),
                );
            }
        }
        let views = tensors
            .iter()
            .map(|(name, (dtype, shape, bytes))| {
                Ok((
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, None, &directory.join(OPTIMIZER_STATE))?;
        let metadata = OptimizerMetadata {
            math_policy:self.math_policy,
            schema: OPTIMIZER_SCHEMA,
            step: self.step,
            config: self.config,
            source_sha256: self.source_sha256.clone(),
            targets: self.targets().map(str::to_owned).collect(),
            adapter_snapshot_sha256: snapshot,
            adapter_file_sha256,
            config_file_sha256,
            optimizer_file_sha256: hash_file(&directory.join(OPTIMIZER_STATE))?,
        };
        fs::write(
            directory.join(OPTIMIZER_METADATA),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }

    /// Resume from `save`, checking that the checkpoint belongs to `adapter`.
    pub fn resume(adapter: &DeepseekV41LoraAdapter, directory: impl AsRef<Path>) -> Result<Self> {
        use safetensors::tensor::Dtype;
        let directory = directory.as_ref();
        let metadata: OptimizerMetadata = serde_json::from_slice(
            &fs::read(directory.join(OPTIMIZER_METADATA))
                .context("V4.1 adapter optimizer metadata is missing")?,
        )?;
        if !matches!(metadata.schema, 2 | 3 | OPTIMIZER_SCHEMA) {
            bail!(
                "unsupported V4.1 adapter optimizer schema {}",
                metadata.schema
            );
        }
        if metadata.schema == 2 && metadata.step > i32::MAX as u64 {
            bail!("V4.1 adapter optimizer step exceeds bias-correction range");
        }
        if adapter.snapshot_digest()? != metadata.adapter_snapshot_sha256
            || hash_file(&directory.join(super::peft::DEEPSEEK_V41_LORA_WEIGHTS))?
                != metadata.adapter_file_sha256
            || hash_file(&directory.join(super::peft::DEEPSEEK_V41_LORA_CONFIG))?
                != metadata.config_file_sha256
            || hash_file(&directory.join(OPTIMIZER_STATE))? != metadata.optimizer_file_sha256
        {
            bail!("V4.1 optimizer checkpoint factors/config/state checksum mismatch");
        }
        let mut trainer = Self::new(adapter, metadata.config)?;
        trainer.math_policy=metadata.math_policy;
        // Keep the original adapter checksum as provenance, while the new
        // snapshot/file bindings above identify the actual resume state.
        let mut expected = metadata.targets.clone();
        let mut actual = trainer.targets().map(str::to_owned).collect::<Vec<_>>();
        expected.sort();
        actual.sort();
        if expected != actual {
            bail!(
                "V4.1 adapter optimizer checkpoint targets {} do not match the adapter's {}",
                expected.join(", "),
                actual.join(", ")
            );
        }
        trainer.source_sha256 = metadata.source_sha256;
        let bytes = fs::read(directory.join(OPTIMIZER_STATE))
            .context("V4.1 adapter optimizer state is missing")?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        if metadata.schema < 4 && tensors.len() != trainer.slots.len() * if metadata.schema == 2 { 4 } else { 6 } {
            bail!("V4.1 optimizer checkpoint has unexpected tensors");
        }
        let read = |name: String, shape: Vec<usize>, square: bool| -> Result<Vec<f32>> {
            let view = tensors
                .tensor(&name)
                .with_context(|| format!("V4.1 adapter optimizer tensor {name} is missing"))?;
            if view.dtype() != Dtype::F32 || view.shape() != shape.as_slice() {
                bail!("V4.1 adapter optimizer tensor {name} has the wrong geometry");
            }
            let values: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            if values
                .iter()
                .any(|v| !v.is_finite() || (square && *v < 0.0))
            {
                bail!("V4.1 optimizer checkpoint tensor {name} has invalid moments");
            }
            Ok(values)
        };
        let read_step = |name: String| -> Result<u64> {
            if metadata.schema == 2 {
                // Schema 2 required every factor's gradient on every call.
                return Ok(metadata.step);
            }
            let view = tensors
                .tensor(&name)
                .with_context(|| format!("V4.1 optimizer counter {name} is missing"))?;
            if view.dtype() != Dtype::U64 || view.shape() != [1] || view.data().len() != 8 {
                bail!("V4.1 optimizer counter {name} has the wrong geometry/type");
            }
            Ok(u64::from_le_bytes(view.data().try_into()?))
        };
        let mut expected_tensors = 0;
        for (target, slot) in &mut trainer.slots {
            let factors = &trainer.factors[target];
            let a_shape = vec![factors.rank, factors.a.len() / factors.rank];
            let b_shape = vec![factors.b.len() / factors.rank, factors.rank];
            slot.step_a = read_step(format!("{target}.step_a"))?;
            slot.step_b = read_step(format!("{target}.step_b"))?;
            if metadata.schema < 4 || slot.step_a > 0 {
                slot.exp_avg_a = read(format!("{target}.exp_avg_a"), a_shape.clone(), false)?;
                slot.exp_avg_sq_a = read(format!("{target}.exp_avg_sq_a"), a_shape, true)?;
            }
            if metadata.schema < 4 || slot.step_b > 0 {
                slot.exp_avg_b = read(format!("{target}.exp_avg_b"), b_shape.clone(), false)?;
                slot.exp_avg_sq_b = read(format!("{target}.exp_avg_sq_b"), b_shape, true)?;
            }
            validate_slot(target, slot, factors, metadata.step)?;
            // Old dense checkpoints may contain the zero moments for parameters
            // which never received a gradient. Validate, then retire them.
            if slot.step_a == 0 { slot.exp_avg_a.clear(); slot.exp_avg_a.shrink_to_fit(); slot.exp_avg_sq_a.clear(); slot.exp_avg_sq_a.shrink_to_fit(); }
            if slot.step_b == 0 { slot.exp_avg_b.clear(); slot.exp_avg_b.shrink_to_fit(); slot.exp_avg_sq_b.clear(); slot.exp_avg_sq_b.shrink_to_fit(); }
            expected_tensors += 2 + 2*usize::from(slot.step_a > 0) + 2*usize::from(slot.step_b > 0);
        }
        if metadata.schema == 4 && tensors.len() != expected_tensors { bail!("V4.1 sparse optimizer checkpoint has unexpected tensors"); }
        trainer.step = metadata.step;
        Ok(trainer)
    }

    /// Publish one complete standard PEFT adapter plus native optimizer state.
    /// Existing destinations are refused and a failed staging run is removed.
    pub fn save_checkpoint(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        if directory.exists() {
            bail!("V4.1 training checkpoint destination already exists");
        }
        let parent = directory
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let staging = parent.join(format!(".v41-training-{}-{nonce}", std::process::id()));
        let mut owns_staging = false;
        let result = (|| -> Result<()> {
            self.adapter()?.export(&staging)?;
            owns_staging = true;
            self.save(&staging)?;
            Self::resume_from_checkpoint(&staging)?;
            fs::rename(&staging, directory).context("publishing V4.1 training checkpoint")?;
            Ok(())
        })();
        if result.is_err() && owns_staging {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    pub fn resume_from_checkpoint(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        Self::resume(&DeepseekV41LoraAdapter::load(directory)?, directory)
    }

    /// Factor values for a target, `(a, b)`, for qualification and inspection.
    pub fn factors(&self, target: &str) -> Option<(&[f32], &[f32])> {
        self.factors
            .get(target)
            .map(|factors| (factors.a.as_slice(), factors.b.as_slice()))
    }

    /// Moment values for a target, `(exp_avg_a, exp_avg_sq_a, exp_avg_b, exp_avg_sq_b)`.
    /// A factor with update count zero has empty slices: no moment payload is
    /// allocated until its first gradient, including an explicit zero gradient.
    pub fn moments(&self, target: &str) -> Option<(&[f32], &[f32], &[f32], &[f32])> {
        self.slots.get(target).map(|slot| {
            (
                slot.exp_avg_a.as_slice(),
                slot.exp_avg_sq_a.as_slice(),
                slot.exp_avg_b.as_slice(),
                slot.exp_avg_sq_b.as_slice(),
            )
        })
    }

    pub fn allocated_moment_elements(&self) -> usize {
        self.slots.values().map(|s|s.exp_avg_a.len()+s.exp_avg_sq_a.len()+s.exp_avg_b.len()+s.exp_avg_sq_b.len()).sum()
    }
}

fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path)
        .with_context(|| format!("reading V4.1 checkpoint {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut chunk = [0u8; 65_536];
    loop {
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        hash.update(&chunk[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn write_bits(values: &mut [f32], bits: &[u32], target: &str, label: &str) -> Result<()> {
    if values.len() != bits.len() {
        bail!("V4.1 adapter state for {target} {label} has the wrong geometry");
    }
    for (value, &raw) in values.iter_mut().zip(bits) {
        let decoded = f32::from_bits(raw);
        if !decoded.is_finite() {
            bail!("V4.1 adapter state for {target} {label} is not finite");
        }
        *value = decoded;
    }
    Ok(())
}

/// Recompute one parameter's optimizer step from raw bits with the pinned
/// single-tensor policy the trainer uses.
///
/// Trajectory gates compare consecutive steps, so a factor difference can be
/// either an optimizer defect or the propagation of an already-gated gradient
/// difference through `eps`. Reproducing the frozen reference state from the
/// frozen gradients separates the two: the update arithmetic is exact when this
/// returns the reference bits, and the remaining difference is then provably
/// the propagation of the gradient difference alone.
#[cfg(test)]
pub(crate) fn reference_adamw_parameter_bits(
    param_bits: &[u32],
    gradient_bits: &[u32],
    avg_bits: &[u32],
    square_bits: &[u32],
    step: u64,
    config: DeepseekV41AdamWConfig,
) -> Vec<u32> {
    let decay = (1.0 - config.learning_rate * config.weight_decay) as f32;
    let step_size = (config.learning_rate / (1.0 - config.beta1.powi(step as i32))) as f32;
    let bias2_sqrt = (1.0 - config.beta2.powi(step as i32)).sqrt() as f32;
    let block = 2 * crate::transformer::aten_vector_width() as usize;
    param_bits
        .iter()
        .enumerate()
        .map(|(index, &raw)| {
            let moment = |bits: &[u32]| bits.get(index).copied().unwrap_or(0);
            let (param, _, _) = update_element(
                f32::from_bits(raw),
                f32::from_bits(gradient_bits[index]),
                f32::from_bits(moment(avg_bits)),
                f32::from_bits(moment(square_bits)),
                config,
                decay,
                step_size,
                bias2_sqrt,
                AdamWMathPolicy::CpuSingleTensorV2,
                index < param_bits.len() / block * block,
            );
            param.to_bits()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn update(
    param: &mut [f32],
    gradient: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    config: DeepseekV41AdamWConfig,
    decay: f32,
    step_size: f32,
    bias_correction2_sqrt: f32,
    policy:AdamWMathPolicy,
) {
    let block=2*crate::transformer::aten_vector_width() as usize;
    for index in 0..param.len() {
        (param[index], exp_avg[index], exp_avg_sq[index]) = update_element(
            param[index],
            gradient[index],
            exp_avg[index],
            exp_avg_sq[index],
            config,
            decay,
            step_size,
            bias_correction2_sqrt,
            policy,
            index<param.len()/block*block,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn update_element(
    param: f32,
    gradient: f32,
    avg: f32,
    square: f32,
    config: DeepseekV41AdamWConfig,
    decay: f32,
    step_size: f32,
    bias_correction2_sqrt: f32,
    policy:AdamWMathPolicy,
    vector:bool,
) -> (f32, f32, f32) {
    let value = param * decay;
    if policy==AdamWMathPolicy::LegacyF32V1 {
        let b1=config.beta1 as f32;let b2=config.beta2 as f32;
        let avg=avg*b1+(1.0-b1)*gradient;
        let square=square*b2+((1.0-b2)*gradient)*gradient;
        let denom=square.sqrt()/bias_correction2_sqrt+config.eps as f32;
        return (value-step_size*(avg/denom),avg,square);
    }
    let weight=(1.0-config.beta1) as f32;
    let difference=gradient-avg;
    let avg=if vector {
        if weight.abs()<0.5 {weight.mul_add(difference,avg)} else {(weight-1.0).mul_add(difference,gradient)}
    } else if weight.abs()<0.5 {avg+weight*difference} else {gradient-difference*(1.0-weight)};
    let square=square*(config.beta2 as f32)+(((1.0-config.beta2) as f32)*gradient)*gradient;
    let denom=square.sqrt()/bias_correction2_sqrt+config.eps as f32;
    // ATen addcdiv evaluates value * numerator before its division.
    (value+((-step_size)*avg)/denom,avg,square)
}

fn validate_slot(
    target: &str,
    slot: &AdamWSlot,
    factors: &TrainableFactors,
    calls: u64,
) -> Result<()> {
    for (name, step, avg, sq, len) in [
        (
            "A",
            slot.step_a,
            &slot.exp_avg_a,
            &slot.exp_avg_sq_a,
            factors.a.len(),
        ),
        (
            "B",
            slot.step_b,
            &slot.exp_avg_b,
            &slot.exp_avg_sq_b,
            factors.b.len(),
        ),
    ] {
        if step > calls
            || step > i32::MAX as u64
            || (avg.len() != len && !(step == 0 && avg.is_empty()))
            || (sq.len() != len && !(step == 0 && sq.is_empty()))
            || avg.iter().chain(sq).any(|v| !v.is_finite())
            || sq.iter().any(|v| *v < 0.0)
            || (step == 0 && avg.iter().chain(sq).any(|v| *v != 0.0))
        {
            bail!("V4.1 optimizer {target} {name} has invalid moments/counter");
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "train_conditional_tests.rs"]
pub(crate) mod conditional_tests;
