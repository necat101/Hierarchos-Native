#![recursion_limit = "256"]

use std::{
    collections::HashSet,
    ffi::OsString,
    fmt::Write as FmtWrite,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    ops::Range,
    path::{Path, PathBuf},
    sync::{mpsc, Arc, OnceLock},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use hierarchos_inference::ModelConfig;
use hierarchos_vulkan::{
    encode_portable_replay_json, encode_portable_running_carriers, read_model_ltm_running_state,
    read_portable_replay_json_field, read_portable_running_carriers, AdamWHyperParams,
    HierarchosBudgetedTokenTapeTrainResult, HierarchosDataStreamCursorState,
    HierarchosExecutionPolicyState, HierarchosFullModelReplicaTransportSource,
    HierarchosJointTrainingAdapterConstraint, HierarchosLabeledSequenceObjective,
    HierarchosLearningRateScheduleState, HierarchosLossScalingState,
    HierarchosPendingGradientTransportSource, HierarchosPortableLtmRunningState,
    HierarchosPortableReplayTensor, HierarchosPortableRunningCarriers,
    HierarchosPortableTrainingReplay, HierarchosRawTokenLabeledSequenceInput,
    HierarchosReplicaStateDeviceGroupTransport, HierarchosReplicaStateRangeRetirement,
    HierarchosReplicaStateRetirementTimeline, HierarchosReplicaStateStreamStats,
    HierarchosReplicaStateTimelineReservation, HierarchosStochasticRngPolicyState,
    HierarchosTapeMemoryPolicy, HierarchosTokenTapeReadbackPolicy, HierarchosTokenTapeUpdateMode,
    HierarchosTrainingCheckpointManifest, HierarchosTrainingGraph,
    HierarchosTrainingPrecisionPolicy, HierarchosTrainingSessionState, VulkanDevice,
    VulkanGradientTransportBackend, VulkanPhysicalDeviceInfo, VulkanSubmissionArenaStats,
    HIERARCHOS_VULKAN_DATA_STREAM_CURSOR_FORMAT, HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT,
    HIERARCHOS_VULKAN_PORTABLE_REPLAY_FORMAT, HIERARCHOS_VULKAN_PORTABLE_SAMPLER_RNG_ALGORITHM,
    HIERARCHOS_VULKAN_TRAINING_MANIFEST_FILENAME, HIERARCHOS_VULKAN_TRAINING_PRECISION_ENV,
    HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT,
};
use safetensors::tensor::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bound the live cross-adapter gradient payload independently of model size.
/// The transport owns one source host-visible window plus bounded destination
/// Vulkan scratch/upload buffers; canonical parameter layout remains FP32 so
/// checkpoints stay interchangeable with PyTorch/CUDA.
const DEFAULT_MULTI_DEVICE_GRADIENT_STREAM_CHUNK_VALUES: usize = 256 * 1024;
/// Build the phase-memory probe with a deliberately small optimizer transport
/// arena. The joint planner projects wider candidates without allocating them,
/// then the probe is dropped and the production graphs are built once at the
/// selected width.
const JOINT_MEMORY_PREFLIGHT_CHUNK_VALUES: usize = 4 * 1024;
const DEFAULT_JOINT_RUNTIME_AUTOTUNE_EXPLORE_EVERY: u64 = 16;
const HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_DISABLE_ENV: &str =
    "HIERARCHOS_VULKAN_DISABLE_JOINT_RUNTIME_AUTOTUNE";
const HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_LOG_ENV: &str =
    "HIERARCHOS_VULKAN_JOINT_RUNTIME_AUTOTUNE_LOG";
const HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_EXPLORE_EVERY_ENV: &str =
    "HIERARCHOS_VULKAN_JOINT_RUNTIME_AUTOTUNE_EXPLORE_EVERY";
const JOINT_RUNTIME_PROFILE_SCHEMA_VERSION: u32 = 1;
const JOINT_RUNTIME_PROFILE_FILENAME: &str = "vulkan_joint_runtime_profile.v1.json";
const JOINT_RUNTIME_PROFILE_LIVE_PERSIST_EVERY_SCORED_WINDOWS: u64 = 8;
const JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS: u64 = 2;
const JOINT_RUNTIME_CONFIDENCE_Z: f64 = 1.645;
/// Exclude two complete optimizer windows after a composite schedule switch.
/// The first absorbs the actual switch; the second lets clocks, descriptor/
/// transport reuse, and any outstanding broadcast retirement settle before a
/// window is admitted to the persisted selector evidence.
const JOINT_RUNTIME_WARMUP_WINDOWS_AFTER_SWITCH: u8 = 2;
/// Conservative throttling detector. A window is only flagged when the same
/// arm already has a stable baseline, effective throughput falls materially,
/// and device-side timestamp cost rises in the opposite direction. Requiring
/// both signals avoids treating host scheduling noise as GPU throttling.
const JOINT_RUNTIME_THROTTLE_THROUGHPUT_RATIO: f64 = 0.80;
const JOINT_RUNTIME_THROTTLE_GPU_COST_RATIO: f64 = 1.20;
const JOINT_RUNTIME_HIGH_MEMORY_PRESSURE_BUCKET: u8 = 6;
// Match the token-tape scheduler's long-run policy: evidence is discounted by
// later scored observations so a once-good arm becomes uncertain again when it
// has not been measured for a while. With the default exploration cadence,
// 0.90 is a roughly 6.6-observation half-life.
const JOINT_RUNTIME_OBSERVATION_DECAY: f64 = 0.90;
const JOINT_RUNTIME_UCB_EXPLORATION_SCALE: f64 = 0.10;
const JOINT_RUNTIME_RELATIVE_NOISE_FLOOR: f64 = 0.10;
const JOINT_RUNTIME_MIN_EFFECTIVE_SAMPLES: f64 = 0.25;
/// Per-device workload shares are learned from the same persisted lane telemetry
/// as the joint arm score. Keep early observations conservative so one noisy
/// window cannot starve a device, but allow a stable heterogeneous topology to
/// converge to a meaningfully asymmetric split.
const JOINT_RUNTIME_DEVICE_SHARE_CONFIDENCE_WINDOWS: f64 = 4.0;
const JOINT_RUNTIME_DEVICE_SHARE_MIN_RELATIVE_CAPACITY: f64 = 0.125;
const JOINT_RUNTIME_DEVICE_SHARE_MAX_RELATIVE_CAPACITY: f64 = 8.0;
/// The native trainer has a deliberately wide scheduler frame: phase-aware
/// memory planning, replica transport, persistent recurrent carriers, and
/// checkpoint state all coexist in one orchestration routine. Windows gives
/// the process main thread a comparatively small fixed stack, which can be
/// exhausted by the debug build before argument parsing begins. Keep the OS
/// entrypoint tiny and run the scheduler on an explicitly sized host stack;
/// Vulkan tensor/optimizer storage remains device- or heap-resident.
const TRAINER_HOST_STACK_BYTES: usize = 32 * 1024 * 1024;
const NATIVE_RUN_IDENTITY_FORMAT: &str = "hierarchos-vulkan-native-run-identity-v1";
const NATIVE_DATASET_HASH_ALGORITHM: &str = "hierarchos-native-jsonl-record-stream-v1";
const PORTABLE_TOKEN_CACHE_INDEX_FILENAME: &str = "index.safetensors";
const PORTABLE_TOKEN_CACHE_INDEX_FORMAT: &str = "hierarchos-token-cache-index-v1";
const TOKEN_CACHE_RECORD_HASH_ALGORITHM: &str = "record-stream-v1";
const TOKEN_CACHE_RECORD_HASH_HEADER: &[u8] = b"hierarchos-token-cache-record-stream-v1\0";
const NATIVE_GUI_EVENT_PREFIX: &str = "HIERARCHOS_EVENT ";
const COHERENT_V9_DEFAULT_TBPTT_CHUNK_SIZE: usize = 256;
const LEGACY_V8_DEFAULT_TBPTT_CHUNK_SIZE: usize = 128;

fn architecture_default_tbptt_chunk_size(architecture_revision: &str) -> usize {
    if architecture_revision.eq_ignore_ascii_case("coherent-v9") {
        COHERENT_V9_DEFAULT_TBPTT_CHUNK_SIZE
    } else {
        LEGACY_V8_DEFAULT_TBPTT_CHUNK_SIZE
    }
}

fn emit_gui_event(enabled: bool, event: serde_json::Value) {
    if !enabled {
        return;
    }
    match serde_json::to_string(&event) {
        Ok(event) => eprintln!("{NATIVE_GUI_EVENT_PREFIX}{event}"),
        Err(err) => eprintln!("WARNING: failed to encode native GUI event: {err}"),
    }
}

fn finite_mean(values: &[f32]) -> Option<f64> {
    let mut total = 0.0f64;
    let mut count = 0u64;
    for &value in values {
        if value.is_finite() {
            total += f64::from(value);
            count += 1;
        }
    }
    (count > 0).then_some(total / count as f64)
}

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    resume_from_checkpoint: bool,
    dataset_path: PathBuf,
    output_dir: PathBuf,
    joint_runtime_profile: Option<PathBuf>,
    lock_joint_runtime_profile: bool,
    epochs: u64,
    batch_size: usize,
    gradient_accumulation_steps: usize,
    learning_rate: f32,
    min_learning_rate: f32,
    grad_clip: f32,
    initial_loss_scale: f64,
    training_precision_override: Option<String>,
    warmup_steps: u64,
    warmup_ratio: f64,
    disable_lr_schedule: bool,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    z_loss_weight: f32,
    ponder_loss_weight: f32,
    commitment_loss_weight: f32,
    max_ce_loss_for_backward: f32,
    max_ponder_cost_for_backward: f32,
    max_commitment_cost_for_backward: f32,
    max_skipped_train_batches: u64,
    save_steps: u64,
    tbptt_chunk_size: Option<usize>,
    trainable_prefixes: Vec<String>,
    persist_state: bool,
    shuffle: bool,
    seed: u64,
    device_index: Option<usize>,
    device_indices: Option<Vec<usize>>,
    gradient_stream_chunk_values: usize,
    json_events: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DatasetRow {
    input_ids: Vec<u32>,
    #[serde(default)]
    labels: Option<Vec<i64>>,
    #[serde(default)]
    attention_mask: Option<Vec<f32>>,
    #[serde(default)]
    loss_weights: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct DatasetSourceIdentity {
    source_kind: String,
    replay_guarantee: String,
    total_tokens: u64,
    token_cache: serde_json::Value,
}

#[derive(Debug)]
struct LoadedDataset {
    rows: Vec<DatasetRow>,
    identity: DatasetSourceIdentity,
}

#[derive(Debug)]
struct PackedBatch {
    tokens: usize,
    input_ids: Vec<u32>,
    labels: Vec<i64>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

fn parse_value<T>(raw: OsString, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    raw.to_string_lossy()
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value for {flag}: {err}"))
}

fn required_value(args: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<OsString> {
    args.next()
        .with_context(|| format!("missing value after {flag}"))
}

fn parse_device_indices(raw: OsString) -> Result<Vec<usize>> {
    let raw = raw.to_string_lossy();
    let mut indices = Vec::new();
    for (position, value) in raw.split(',').enumerate() {
        let value = value.trim();
        anyhow::ensure!(
            !value.is_empty(),
            "--device-indices contains an empty entry at position {}",
            position + 1
        );
        indices.push(value.parse::<usize>().with_context(|| {
            format!("invalid Vulkan device index {value:?} in --device-indices")
        })?);
    }
    anyhow::ensure!(
        !indices.is_empty(),
        "--device-indices requires at least one Vulkan device index"
    );
    Ok(indices)
}

fn parse_training_precision(raw: OsString) -> Result<String> {
    let value = raw.to_string_lossy();
    let canonical = match value.trim().to_ascii_lowercase().as_str() {
        "fp32" => "fp32",
        "fp16" | "fp16-storage-fp32-compute" => "fp16-storage-fp32-compute",
        "fp16-parity" | "fp16-storage-parity" => "fp16-storage-parity",
        "fp16-lm-backward" | "fp16-storage-fp16-lm-backward" => {
            "fp16-storage-fp16-lm-backward"
        }
        other => bail!(
            "unsupported --precision {other:?}; expected fp32, fp16-storage-fp32-compute, fp16-storage-parity, or fp16-storage-fp16-lm-backward"
        ),
    };
    Ok(canonical.to_owned())
}

fn resolve_training_source(
    model_dir: Option<PathBuf>,
    resume_from_ckpt: Option<PathBuf>,
) -> Result<(PathBuf, bool)> {
    match (model_dir, resume_from_ckpt) {
        (Some(model_dir), None) => Ok((model_dir, false)),
        (None, Some(resume_from_ckpt)) => Ok((resume_from_ckpt, true)),
        (Some(_), Some(_)) => bail!(
            "--model and --resume-from-ckpt are mutually exclusive: use --model for a fresh optimizer/session over package weights, or --resume-from-ckpt for exact training continuation"
        ),
        (None, None) => bail!(
            "missing training source: supply --model MODEL_DIR for fresh training or --resume-from-ckpt PACKAGE_DIR for exact continuation"
        ),
    }
}

fn require_package_local_file(package_dir: &Path, member: &str, label: &str) -> Result<PathBuf> {
    anyhow::ensure!(
        !member.is_empty() && Path::new(member).components().count() == 1,
        "{label} must be a package-local filename, got {member:?}"
    );
    let path = package_dir.join(member);
    anyhow::ensure!(
        path.is_file(),
        "{label} is missing from exact-resume package: {}",
        path.display()
    );
    Ok(path)
}

fn validate_exact_resume_package(package_dir: &Path) -> Result<()> {
    let manifest_path = package_dir.join(HIERARCHOS_VULKAN_TRAINING_MANIFEST_FILENAME);
    anyhow::ensure!(
        manifest_path.is_file(),
        "--resume-from-ckpt requires a training checkpoint manifest at {}; use --model instead for a weights-only fresh run",
        manifest_path.display()
    );
    let manifest: HierarchosTrainingCheckpointManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?,
    )
    .with_context(|| format!("decoding {}", manifest_path.display()))?;

    let session = manifest.training_session.as_ref().context(
        "--resume-from-ckpt requires backend-neutral training_session metadata; use --model for a weights-only fresh run",
    )?;
    anyhow::ensure!(
        session.data_stream_cursor.is_some(),
        "--resume-from-ckpt requires an exact data_stream_cursor in training_session"
    );
    anyhow::ensure!(
        session.execution_policy.is_some(),
        "--resume-from-ckpt requires an exact execution_policy in training_session"
    );

    require_package_local_file(package_dir, &manifest.model_file, "checkpoint model_file")?;
    require_package_local_file(
        package_dir,
        &manifest.optimizer_file,
        "checkpoint optimizer_file",
    )?;
    if manifest.accumulation_open {
        let gradient_file = manifest
            .gradient_file
            .as_deref()
            .context("--resume-from-ckpt open accumulation window is missing gradient_file")?;
        require_package_local_file(package_dir, gradient_file, "checkpoint gradient_file")?;
    }

    let replay_file = manifest.portable_replay_file.as_deref().context(
        "--resume-from-ckpt requires a portable replay sidecar; use --model instead for a weights-only fresh run",
    )?;
    let replay_path =
        require_package_local_file(package_dir, replay_file, "checkpoint portable_replay_file")?;
    let replay_document: serde_json::Value = serde_json::from_slice(
        &fs::read(&replay_path).with_context(|| format!("reading {}", replay_path.display()))?,
    )
    .with_context(|| format!("decoding {}", replay_path.display()))?;
    anyhow::ensure!(
        replay_document
            .get("format")
            .and_then(serde_json::Value::as_str)
            == Some(HIERARCHOS_VULKAN_PORTABLE_REPLAY_FORMAT),
        "portable replay sidecar {} has unsupported format {:?}",
        replay_path.display(),
        replay_document.get("format")
    );
    anyhow::ensure!(
        replay_document.get("state").is_some(),
        "portable replay sidecar {} is missing state",
        replay_path.display()
    );

    if let Some(replay_tensor_file) = manifest.portable_replay_tensor_file.as_deref() {
        require_package_local_file(
            package_dir,
            replay_tensor_file,
            "checkpoint portable_replay_tensor_file",
        )?;
    }
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut model_dir = None;
    let mut resume_from_ckpt = None;
    let mut dataset_path = None;
    let mut output_dir = None;
    let mut joint_runtime_profile = None;
    let mut lock_joint_runtime_profile = false;
    let mut epochs = 1u64;
    let mut batch_size = 1usize;
    let mut gradient_accumulation_steps = 1usize;
    let mut learning_rate = 1.0e-4f32;
    let mut min_learning_rate = 0.0f32;
    let mut grad_clip = 1.0f32;
    let mut initial_loss_scale = 65_536.0f64;
    let mut training_precision_override = None;
    let mut warmup_steps = 0u64;
    let mut warmup_ratio = 0.0f64;
    let mut disable_lr_schedule = false;
    let mut beta1 = 0.9f32;
    let mut beta2 = 0.999f32;
    let mut eps = 1.0e-8f32;
    let mut weight_decay = 0.1f32;
    let mut z_loss_weight = 1.0e-4f32;
    let mut ponder_loss_weight = 0.003f32;
    let mut commitment_loss_weight = 0.5f32;
    let mut max_ce_loss_for_backward = 0.0f32;
    let mut max_ponder_cost_for_backward = 0.0f32;
    let mut max_commitment_cost_for_backward = 2.0f32;
    let mut max_skipped_train_batches = 0u64;
    let mut save_steps = 0u64;
    let mut tbptt_chunk_size = None;
    let mut trainable_prefixes = Vec::new();
    let mut persist_state = false;
    let mut shuffle = true;
    let mut seed = 0u64;
    let mut device_index = None;
    let mut device_indices = None;
    let mut gradient_stream_chunk_values = DEFAULT_MULTI_DEVICE_GRADIENT_STREAM_CHUNK_VALUES;
    let mut json_events = false;

    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model_dir = Some(PathBuf::from(required_value(&mut args, "--model")?)),
            "--resume-from-ckpt" => {
                resume_from_ckpt = Some(PathBuf::from(required_value(
                    &mut args,
                    "--resume-from-ckpt",
                )?))
            }
            "--dataset" => {
                dataset_path = Some(PathBuf::from(required_value(&mut args, "--dataset")?))
            }
            "--output" => output_dir = Some(PathBuf::from(required_value(&mut args, "--output")?)),
            "--joint-runtime-profile" => {
                joint_runtime_profile = Some(PathBuf::from(required_value(
                    &mut args,
                    "--joint-runtime-profile",
                )?))
            }
            "--lock-joint-runtime-profile" => lock_joint_runtime_profile = true,
            "--epochs" => epochs = parse_value(required_value(&mut args, "--epochs")?, "--epochs")?,
            "--batch-size" => {
                batch_size =
                    parse_value(required_value(&mut args, "--batch-size")?, "--batch-size")?
            }
            "--gradient-accumulation-steps" => {
                gradient_accumulation_steps = parse_value(
                    required_value(&mut args, "--gradient-accumulation-steps")?,
                    "--gradient-accumulation-steps",
                )?
            }
            "--lr" => learning_rate = parse_value(required_value(&mut args, "--lr")?, "--lr")?,
            "--min-lr" => {
                min_learning_rate = parse_value(required_value(&mut args, "--min-lr")?, "--min-lr")?
            }
            "--grad-clip" => {
                grad_clip = parse_value(required_value(&mut args, "--grad-clip")?, "--grad-clip")?
            }
            "--initial-loss-scale" => {
                initial_loss_scale = parse_value(
                    required_value(&mut args, "--initial-loss-scale")?,
                    "--initial-loss-scale",
                )?
            }
            "--precision" => {
                training_precision_override = Some(parse_training_precision(required_value(
                    &mut args,
                    "--precision",
                )?)?)
            }
            "--warmup-steps" => {
                warmup_steps = parse_value(
                    required_value(&mut args, "--warmup-steps")?,
                    "--warmup-steps",
                )?
            }
            "--warmup-ratio" => {
                warmup_ratio = parse_value(
                    required_value(&mut args, "--warmup-ratio")?,
                    "--warmup-ratio",
                )?
            }
            "--disable-lr-schedule" => disable_lr_schedule = true,
            "--beta1" => beta1 = parse_value(required_value(&mut args, "--beta1")?, "--beta1")?,
            "--beta2" => beta2 = parse_value(required_value(&mut args, "--beta2")?, "--beta2")?,
            "--eps" => eps = parse_value(required_value(&mut args, "--eps")?, "--eps")?,
            "--weight-decay" => {
                weight_decay = parse_value(
                    required_value(&mut args, "--weight-decay")?,
                    "--weight-decay",
                )?
            }
            "--z-loss-weight" => {
                z_loss_weight = parse_value(
                    required_value(&mut args, "--z-loss-weight")?,
                    "--z-loss-weight",
                )?
            }
            "--ponder-loss-weight" => {
                ponder_loss_weight = parse_value(
                    required_value(&mut args, "--ponder-loss-weight")?,
                    "--ponder-loss-weight",
                )?
            }
            "--commitment-loss-weight" => {
                commitment_loss_weight = parse_value(
                    required_value(&mut args, "--commitment-loss-weight")?,
                    "--commitment-loss-weight",
                )?
            }
            "--max-ce-loss-for-backward" => {
                max_ce_loss_for_backward = parse_value(
                    required_value(&mut args, "--max-ce-loss-for-backward")?,
                    "--max-ce-loss-for-backward",
                )?
            }
            "--max-ponder-cost-for-backward" => {
                max_ponder_cost_for_backward = parse_value(
                    required_value(&mut args, "--max-ponder-cost-for-backward")?,
                    "--max-ponder-cost-for-backward",
                )?
            }
            "--max-commitment-cost-for-backward" => {
                max_commitment_cost_for_backward = parse_value(
                    required_value(&mut args, "--max-commitment-cost-for-backward")?,
                    "--max-commitment-cost-for-backward",
                )?
            }
            "--max-skipped-train-batches" => {
                max_skipped_train_batches = parse_value(
                    required_value(&mut args, "--max-skipped-train-batches")?,
                    "--max-skipped-train-batches",
                )?
            }
            "--save-steps" => {
                save_steps =
                    parse_value(required_value(&mut args, "--save-steps")?, "--save-steps")?
            }
            "--tbptt-chunk-size" => {
                tbptt_chunk_size = Some(parse_value(
                    required_value(&mut args, "--tbptt-chunk-size")?,
                    "--tbptt-chunk-size",
                )?)
            }
            "--trainable-prefix" => trainable_prefixes.push(
                required_value(&mut args, "--trainable-prefix")?
                    .to_string_lossy()
                    .into_owned(),
            ),
            "--seed" => seed = parse_value(required_value(&mut args, "--seed")?, "--seed")?,
            "--device-index" => {
                device_index = Some(parse_value(
                    required_value(&mut args, "--device-index")?,
                    "--device-index",
                )?)
            }
            "--device-indices" => {
                device_indices = Some(parse_device_indices(required_value(
                    &mut args,
                    "--device-indices",
                )?)?)
            }
            "--gradient-stream-chunk-values" => {
                gradient_stream_chunk_values = parse_value(
                    required_value(&mut args, "--gradient-stream-chunk-values")?,
                    "--gradient-stream-chunk-values",
                )?
            }
            "--persist-state" => persist_state = true,
            "--no-shuffle" => shuffle = false,
            "--json-events" => json_events = true,
            "--help" | "-h" => {
                println!(
                    "hierarchos-vulkan-train (--model MODEL_DIR | --resume-from-ckpt PACKAGE_DIR) --dataset TOKENS.jsonl|TOKEN_CACHE_DIR --output OUTPUT_DIR \
                     [--epochs N] [--batch-size N] [--gradient-accumulation-steps N] \
                     [--lr F] [--min-lr F] [--grad-clip F] [--precision POLICY] [--initial-loss-scale F] [--warmup-steps N] [--warmup-ratio F] \
                     [--disable-lr-schedule] [--beta1 F] [--beta2 F] [--eps F] [--weight-decay F] \
                     [--z-loss-weight F] [--ponder-loss-weight F] [--commitment-loss-weight F] \
                     [--max-ce-loss-for-backward F] [--max-ponder-cost-for-backward F] \
                     [--max-commitment-cost-for-backward F] [--max-skipped-train-batches N] [--save-steps N] \
                     [--tbptt-chunk-size N] [--device-index N | --device-indices N,N,...] \
                     [--trainable-prefix CANONICAL_NAME_PREFIX]... \
                     [--gradient-stream-chunk-values N] \
                     [--joint-runtime-profile vulkan_joint_runtime_profile.v1.json] \
                     [--lock-joint-runtime-profile] \
                     [--persist-state] [--seed N] [--no-shuffle] [--json-events]\n\n\
                     --dataset accepts legacy tokenized JSONL or a Hierarchos schema-v6 token-cache directory.\n\
                     Token-cache directories are the cross-runtime parity path: Python/PyTorch and native Vulkan consume the same content-addressed records through index.safetensors + tokens.bin.\n\
                     JSONL rows require input_ids and may contain labels, attention_mask, and loss_weights. \
                     Missing labels default to input_ids, preserving the canonical next-token shift.\n\
                     --model always starts a fresh optimizer/session from the package weights, even when that directory also contains training_state.json.\n\
                     --resume-from-ckpt restores model weights, AdamW/pending gradients, scheduler/scaler, data cursor, and portable replay state from an exact-resume package.\n\
                     --precision selects fp32, fp16-storage-fp32-compute, fp16-storage-parity, or fp16-storage-fp16-lm-backward and overrides HIERARCHOS_VULKAN_TRAINING_PRECISION for this trainer process.\n\
                     --initial-loss-scale selects the fresh FP16 GradScaler scale (default 65536); exact resume always restores the checkpoint's live scaler state.\n\
                     --epochs is the total target epoch count when resuming a native training session.\n\
                     --trainable-prefix may be repeated to freeze every optimizer tensor except the selected canonical names/subtrees. Omitting it trains the full model. The selection is exact-resume identity-bound.\n\
                     --device-indices enables synchronous data-parallel Vulkan training across the \
                     listed physical adapters; independent optimizer-window microbatches are sharded \
                     contiguously by token workload, then reduced in logical order on the first adapter.\n\
                     --gradient-stream-chunk-values bounds the FP32 values resident in each cross-adapter \
                     staging window; larger chunks reduce queue submissions at the cost of bounded host-visible memory.\n\
                     --joint-runtime-profile replays a separately collected multi-device scheduler profile only when \
                     its architecture, batch/token geometry, ordered device+driver UUIDs, and transport backends exactly match.\n\
                     --lock-joint-runtime-profile freezes transport width, optimizer/broadcast overlap, and tape geometry \
                     to that profile's winning arm for hardware qualification instead of permitting online exploration.\n\
                     --persist-state carries recurrent/context/ROSA values across consecutive batches and \
                     therefore requires --no-shuffle plus lane-contiguous dataset ordering.\n\
                     --save-steps writes exact-resume Vulkan packages under OUTPUT/checkpoint-epoch-E-step-S. \
                     Persisted mid-epoch checkpoints include the PyTorch-compatible H/L/context/LTM/ROSA carrier.\n\
                     --json-events emits single-line HIERARCHOS_EVENT JSON records on stderr for native GUI supervision."
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; use --help for usage"),
        }
    }

    let (model_dir, resume_from_checkpoint) = resolve_training_source(model_dir, resume_from_ckpt)?;
    let args = Args {
        model_dir,
        resume_from_checkpoint,
        dataset_path: dataset_path
            .context("missing required --dataset TOKENS.jsonl|TOKEN_CACHE_DIR")?,
        output_dir: output_dir.context("missing required --output OUTPUT_DIR")?,
        joint_runtime_profile,
        lock_joint_runtime_profile,
        epochs,
        batch_size,
        gradient_accumulation_steps,
        learning_rate,
        min_learning_rate,
        grad_clip,
        initial_loss_scale,
        training_precision_override,
        warmup_steps,
        warmup_ratio,
        disable_lr_schedule,
        beta1,
        beta2,
        eps,
        weight_decay,
        z_loss_weight,
        ponder_loss_weight,
        commitment_loss_weight,
        max_ce_loss_for_backward,
        max_ponder_cost_for_backward,
        max_commitment_cost_for_backward,
        max_skipped_train_batches,
        save_steps,
        tbptt_chunk_size,
        trainable_prefixes,
        persist_state,
        shuffle,
        seed,
        device_index,
        device_indices,
        gradient_stream_chunk_values,
        json_events,
    };
    validate_args(&args)?;
    Ok(args)
}

fn validate_args(args: &Args) -> Result<()> {
    anyhow::ensure!(args.epochs > 0, "--epochs must be positive");
    anyhow::ensure!(args.batch_size > 0, "--batch-size must be positive");
    anyhow::ensure!(
        args.gradient_accumulation_steps > 0,
        "--gradient-accumulation-steps must be positive"
    );
    anyhow::ensure!(
        args.gradient_stream_chunk_values > 0,
        "--gradient-stream-chunk-values must be positive"
    );
    anyhow::ensure!(
        !(args.device_index.is_some() && args.device_indices.is_some()),
        "--device-index and --device-indices are mutually exclusive"
    );
    if args.joint_runtime_profile.is_some() {
        anyhow::ensure!(
            args.device_indices
                .as_ref()
                .is_some_and(|indices| indices.len() > 1),
            "--joint-runtime-profile requires multi-device training via --device-indices with at least two physical adapters"
        );
    }
    anyhow::ensure!(
        !args.lock_joint_runtime_profile || args.joint_runtime_profile.is_some(),
        "--lock-joint-runtime-profile requires --joint-runtime-profile"
    );
    if let Some(indices) = args.device_indices.as_ref() {
        let unique = indices.iter().copied().collect::<HashSet<_>>();
        anyhow::ensure!(
            unique.len() == indices.len(),
            "--device-indices must not contain duplicate physical-device indices"
        );
        anyhow::ensure!(
            !(args.persist_state && indices.len() > 1),
            "multi-device Vulkan training currently requires independent zero-state optimizer windows; --persist-state is not supported with multiple --device-indices"
        );
    }
    if let Some(chunk) = args.tbptt_chunk_size {
        anyhow::ensure!(chunk > 0, "--tbptt-chunk-size must be positive");
    }
    if !args.trainable_prefixes.is_empty() {
        anyhow::ensure!(
            args.trainable_prefixes
                .iter()
                .all(|prefix| !prefix.trim().is_empty()),
            "--trainable-prefix must not be empty"
        );
        let unique = args
            .trainable_prefixes
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        anyhow::ensure!(
            unique.len() == args.trainable_prefixes.len(),
            "--trainable-prefix must not contain duplicates"
        );
    }
    for (name, value) in [
        ("--lr", args.learning_rate),
        ("--min-lr", args.min_learning_rate),
        ("--grad-clip", args.grad_clip),
        ("--eps", args.eps),
        ("--z-loss-weight", args.z_loss_weight),
        ("--ponder-loss-weight", args.ponder_loss_weight),
        ("--commitment-loss-weight", args.commitment_loss_weight),
        ("--max-ce-loss-for-backward", args.max_ce_loss_for_backward),
        (
            "--max-ponder-cost-for-backward",
            args.max_ponder_cost_for_backward,
        ),
        (
            "--max-commitment-cost-for-backward",
            args.max_commitment_cost_for_backward,
        ),
        ("--weight-decay", args.weight_decay),
    ] {
        anyhow::ensure!(value.is_finite(), "{name} must be finite");
    }
    anyhow::ensure!(args.learning_rate > 0.0, "--lr must be positive");
    anyhow::ensure!(
        args.min_learning_rate >= 0.0 && args.min_learning_rate <= args.learning_rate,
        "--min-lr must be non-negative and no greater than --lr"
    );
    anyhow::ensure!(args.grad_clip >= 0.0, "--grad-clip must be non-negative");
    anyhow::ensure!(
        args.initial_loss_scale.is_finite()
            && args.initial_loss_scale > 0.0
            && args.initial_loss_scale <= f64::from(f32::MAX),
        "--initial-loss-scale must be finite, positive, and representable as f32"
    );
    anyhow::ensure!(
        args.warmup_ratio.is_finite() && (0.0..=1.0).contains(&args.warmup_ratio),
        "--warmup-ratio must be finite and in [0,1]"
    );
    anyhow::ensure!(
        !(args.persist_state && args.shuffle),
        "--persist-state requires --no-shuffle so each batch lane remains a contiguous sequence"
    );
    anyhow::ensure!(args.eps > 0.0, "--eps must be positive");
    anyhow::ensure!(
        args.weight_decay >= 0.0,
        "--weight-decay must be non-negative"
    );
    anyhow::ensure!(
        args.z_loss_weight >= 0.0,
        "--z-loss-weight must be non-negative"
    );
    anyhow::ensure!(
        args.max_ce_loss_for_backward >= 0.0
            && args.max_ponder_cost_for_backward >= 0.0
            && args.max_commitment_cost_for_backward >= 0.0,
        "loss-component backward caps must be non-negative"
    );
    anyhow::ensure!(
        args.beta1 >= 0.0 && args.beta1 < 1.0,
        "--beta1 must be in [0,1)"
    );
    anyhow::ensure!(
        args.beta2 >= 0.0 && args.beta2 < 1.0,
        "--beta2 must be in [0,1)"
    );
    Ok(())
}

fn load_jsonl_dataset_rows(path: &Path) -> Result<Vec<DatasetRow>> {
    let file = File::open(path).with_context(|| format!("opening dataset {}", path.display()))?;
    let mut rows = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading dataset line {}", line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut row: DatasetRow = serde_json::from_str(&line)
            .with_context(|| format!("decoding dataset line {}", line_index + 1))?;
        anyhow::ensure!(
            !row.input_ids.is_empty(),
            "dataset line {} has empty input_ids",
            line_index + 1
        );
        let tokens = row.input_ids.len();
        match row.labels.as_ref() {
            Some(labels) => anyhow::ensure!(
                labels.len() == tokens,
                "dataset line {} labels must have the same length as input_ids for the native trainer MVP",
                line_index + 1
            ),
            None => row.labels = Some(row.input_ids.iter().map(|&token| i64::from(token)).collect()),
        }
        if let Some(mask) = row.attention_mask.as_ref() {
            anyhow::ensure!(
                mask.len() == tokens && mask.iter().all(|value| value.is_finite() && *value >= 0.0),
                "dataset line {} attention_mask must match input_ids and contain finite non-negative values",
                line_index + 1
            );
        }
        if let Some(weights) = row.loss_weights.as_ref() {
            anyhow::ensure!(
                weights.len() == tokens
                    && weights.iter().all(|value| value.is_finite() && *value >= 0.0),
                "dataset line {} loss_weights must match input_ids and contain finite non-negative values",
                line_index + 1
            );
        }
        rows.push(row);
    }
    anyhow::ensure!(!rows.is_empty(), "dataset contains no training rows");
    Ok(rows)
}

fn dataset_row_lengths(dataset: &[DatasetRow]) -> Result<Vec<u64>> {
    dataset
        .iter()
        .map(|row| {
            u64::try_from(row.input_ids.len())
                .context("dataset row token length exceeds portable cursor range")
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing SHA-256 hex into String");
    }
    encoded
}

fn normalized_dataset_identity(dataset: &[DatasetRow]) -> Result<(String, u64)> {
    let mut hasher = Sha256::new();
    hasher.update(NATIVE_DATASET_HASH_ALGORITHM.as_bytes());
    hasher.update([0]);
    hasher.update(
        u64::try_from(dataset.len())
            .context("dataset row count exceeds portable identity range")?
            .to_le_bytes(),
    );
    let mut total_tokens = 0u64;
    for row in dataset {
        let token_count = u64::try_from(row.input_ids.len())
            .context("dataset row length exceeds portable identity range")?;
        total_tokens = total_tokens
            .checked_add(token_count)
            .context("dataset token count overflow")?;
        hasher.update(token_count.to_le_bytes());
        for &token in &row.input_ids {
            hasher.update(token.to_le_bytes());
        }
        match row.labels.as_ref() {
            Some(labels) => {
                anyhow::ensure!(
                    labels.len() == row.input_ids.len(),
                    "dataset identity observed labels with incompatible length"
                );
                for &label in labels {
                    hasher.update(label.to_le_bytes());
                }
            }
            None => {
                for &token in &row.input_ids {
                    hasher.update(i64::from(token).to_le_bytes());
                }
            }
        }
        for index in 0..row.input_ids.len() {
            let mask = row
                .attention_mask
                .as_ref()
                .map_or(1.0, |values| values[index]);
            let weight = row
                .loss_weights
                .as_ref()
                .map_or(1.0, |values| values[index]);
            hasher.update(mask.to_bits().to_le_bytes());
            hasher.update(weight.to_bits().to_le_bytes());
        }
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing dataset SHA-256 into String");
    }
    Ok((encoded, total_tokens))
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn read_portable_index_i64(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<i64>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("portable token-cache index is missing tensor {name:?}"))?;
    anyhow::ensure!(
        tensor.shape().len() == 1,
        "portable token-cache index tensor {name:?} must be one-dimensional"
    );
    let raw = tensor.data();
    let mut values = Vec::with_capacity(match tensor.dtype() {
        Dtype::U8 => raw.len(),
        Dtype::I32 | Dtype::U32 => raw.len() / 4,
        Dtype::I64 => raw.len() / 8,
        dtype => bail!(
            "portable token-cache index tensor {name:?} must use U8/I32/U32/I64, got {dtype:?}"
        ),
    });
    match tensor.dtype() {
        Dtype::U8 => values.extend(raw.iter().copied().map(i64::from)),
        Dtype::I32 => {
            anyhow::ensure!(raw.len() % 4 == 0, "invalid I32 byte length for {name:?}");
            values.extend(raw.chunks_exact(4).map(|chunk| {
                i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }));
        }
        Dtype::U32 => {
            anyhow::ensure!(raw.len() % 4 == 0, "invalid U32 byte length for {name:?}");
            values.extend(raw.chunks_exact(4).map(|chunk| {
                i64::from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }));
        }
        Dtype::I64 => {
            anyhow::ensure!(raw.len() % 8 == 0, "invalid I64 byte length for {name:?}");
            values.extend(raw.chunks_exact(8).map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            }));
        }
        _ => unreachable!("portable index dtype validated above"),
    }
    Ok(values)
}

fn decode_cache_input_ids(raw: &[u8], dtype: &str) -> Result<Vec<u32>> {
    match dtype {
        "uint16" => {
            anyhow::ensure!(
                raw.len() % 2 == 0,
                "uint16 token record has an odd byte length"
            );
            Ok(raw
                .chunks_exact(2)
                .map(|chunk| u32::from(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect())
        }
        "int32" => {
            anyhow::ensure!(
                raw.len() % 4 == 0,
                "int32 token record has an invalid byte length"
            );
            raw.chunks_exact(4)
                .map(|chunk| {
                    let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    u32::try_from(value)
                        .with_context(|| format!("token cache contains negative input id {value}"))
                })
                .collect()
        }
        other => bail!("unsupported token-cache token_dtype {other:?}"),
    }
}

fn decode_cache_labels(raw: &[u8], dtype: &str, ignore_sentinel: i64) -> Result<Vec<i64>> {
    match dtype {
        "uint16" => {
            anyhow::ensure!(
                raw.len() % 2 == 0,
                "uint16 label record has an odd byte length"
            );
            Ok(raw
                .chunks_exact(2)
                .map(|chunk| {
                    let value = i64::from(u16::from_le_bytes([chunk[0], chunk[1]]));
                    if value == ignore_sentinel {
                        -100
                    } else {
                        value
                    }
                })
                .collect())
        }
        "int32" => {
            anyhow::ensure!(
                raw.len() % 4 == 0,
                "int32 label record has an invalid byte length"
            );
            raw.chunks_exact(4)
                .map(|chunk| {
                    let value =
                        i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    anyhow::ensure!(
                        value == ignore_sentinel || value >= 0,
                        "token cache contains unsupported negative label {value}"
                    );
                    Ok(if value == ignore_sentinel {
                        -100
                    } else {
                        value
                    })
                })
                .collect()
        }
        other => bail!("unsupported token-cache label_dtype {other:?}"),
    }
}

fn load_token_cache_dataset(cache_dir: &Path) -> Result<LoadedDataset> {
    let success_path = cache_dir.join("_SUCCESS");
    let data_path = cache_dir.join("tokens.bin");
    let success: serde_json::Value = serde_json::from_slice(
        &fs::read(&success_path)
            .with_context(|| format!("reading token-cache manifest {}", success_path.display()))?,
    )
    .with_context(|| format!("decoding token-cache manifest {}", success_path.display()))?;
    let success_object = success
        .as_object()
        .context("token-cache _SUCCESS manifest must be a JSON object")?;

    let schema = success_object
        .get("storage_schema_version")
        .and_then(serde_json::Value::as_u64)
        .context("token-cache _SUCCESS is missing storage_schema_version")?;
    anyhow::ensure!(
        schema == 6,
        "native Vulkan training currently requires Hierarchos token-cache schema v6; rebuild this cache with a current Hierarchos token-cache builder or use tokenized JSONL"
    );
    anyhow::ensure!(
        success_object
            .get("byte_order")
            .and_then(serde_json::Value::as_str)
            == Some("little"),
        "token-cache schema v6 requires byte_order='little'"
    );

    let portable_index_name = success_object
        .get("portable_index_file")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(PORTABLE_TOKEN_CACHE_INDEX_FILENAME);
    anyhow::ensure!(
        Path::new(portable_index_name).components().count() == 1,
        "token-cache portable index must be a cache-local filename"
    );
    anyhow::ensure!(
        success_object
            .get("portable_index_format")
            .and_then(serde_json::Value::as_str)
            == Some(PORTABLE_TOKEN_CACHE_INDEX_FORMAT),
        "token cache does not contain the {PORTABLE_TOKEN_CACHE_INDEX_FORMAT} cross-runtime index; rebuild it with the current Hierarchos Python data pipeline"
    );
    let expected_portable_sha = success_object
        .get("portable_index_sha256")
        .and_then(serde_json::Value::as_str)
        .context("token-cache _SUCCESS is missing portable_index_sha256")?;
    anyhow::ensure!(
        is_sha256_digest(expected_portable_sha),
        "token-cache portable_index_sha256 is malformed"
    );
    let portable_index_path = cache_dir.join(portable_index_name);
    let portable_index_bytes = fs::read(&portable_index_path).with_context(|| {
        format!(
            "reading portable token-cache index {}; rebuild the cache if this file is absent",
            portable_index_path.display()
        )
    })?;
    anyhow::ensure!(
        sha256_hex(&portable_index_bytes).eq_ignore_ascii_case(expected_portable_sha),
        "token-cache portable index checksum failed for {}",
        portable_index_path.display()
    );
    let portable_tensors = SafeTensors::deserialize(&portable_index_bytes)
        .context("decoding portable token-cache SafeTensors index")?;
    let offsets = read_portable_index_i64(&portable_tensors, "offsets")?;
    let lengths = read_portable_index_i64(&portable_tensors, "lengths")?;
    anyhow::ensure!(
        !lengths.is_empty() && offsets.len() == lengths.len(),
        "portable token-cache offsets/lengths are empty or have different sizes"
    );
    anyhow::ensure!(
        offsets.iter().all(|&value| value >= 0) && lengths.iter().all(|&value| value > 0),
        "portable token-cache offsets must be nonnegative and lengths must be positive"
    );

    let samples = usize::try_from(
        success_object
            .get("samples")
            .and_then(serde_json::Value::as_u64)
            .context("token-cache _SUCCESS is missing samples")?,
    )
    .context("token-cache sample count exceeds host usize range")?;
    anyhow::ensure!(
        samples == lengths.len(),
        "token-cache sample count mismatch: _SUCCESS={samples}, portable_index={}",
        lengths.len()
    );

    let token_dtype = success_object
        .get("token_dtype")
        .and_then(serde_json::Value::as_str)
        .context("token-cache _SUCCESS is missing token_dtype")?;
    let token_bytes = match token_dtype {
        "uint16" => 2usize,
        "int32" => 4usize,
        other => bail!("unsupported token-cache token_dtype {other:?}"),
    };
    let label_encoding = success_object
        .get("label_encoding")
        .and_then(serde_json::Value::as_str);
    let labels_alias_input_ids = label_encoding == Some("input_ids_alias");
    anyhow::ensure!(
        label_encoding.is_none() || labels_alias_input_ids,
        "unsupported token-cache label_encoding {:?}",
        label_encoding
    );
    let label_dtype = if labels_alias_input_ids {
        None
    } else {
        Some(
            success_object
                .get("label_dtype")
                .and_then(serde_json::Value::as_str)
                .context("token cache with stored labels is missing label_dtype")?,
        )
    };
    if let Some(label_dtype) = label_dtype {
        anyhow::ensure!(
            label_dtype == token_dtype,
            "native schema-v6 loader requires label_dtype to match token_dtype"
        );
    }
    let label_bytes = if labels_alias_input_ids {
        0
    } else {
        token_bytes
    };
    let label_ignore_sentinel = if labels_alias_input_ids {
        -100i64
    } else {
        success_object
            .get("label_ignore_sentinel")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(if token_dtype == "uint16" { 65535 } else { -100 })
    };
    anyhow::ensure!(
        labels_alias_input_ids
            || (token_dtype == "uint16" && label_ignore_sentinel == 65535)
            || (token_dtype == "int32" && label_ignore_sentinel == -100),
        "token-cache label ignore sentinel is incompatible with its integer dtype"
    );

    let has_rosa_ids = success_object
        .get("has_rosa_ids")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if has_rosa_ids {
        anyhow::ensure!(
            success_object
                .get("rosa_dtype")
                .and_then(serde_json::Value::as_str)
                == Some(token_dtype),
            "native schema-v6 loader requires rosa_dtype to match token_dtype"
        );
    }
    let rosa_bytes = if has_rosa_ids { token_bytes } else { 0 };

    let has_loss_weights = success_object
        .get("has_loss_weights")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let (loss_palette, loss_run_offsets, loss_run_ends, loss_run_codes) =
        if has_loss_weights {
            anyhow::ensure!(
                success_object
                    .get("loss_weight_encoding")
                    .and_then(serde_json::Value::as_str)
                    == Some("float32_palette_rle"),
                "token-cache schema v6 loss weights require float32_palette_rle encoding"
            );
            let palette = success_object
                .get("loss_weight_palette")
                .and_then(serde_json::Value::as_array)
                .context("weighted token cache is missing loss_weight_palette")?
                .iter()
                .map(|value| {
                    let value = value
                        .as_f64()
                        .context("token-cache loss-weight palette contains a non-number")?
                        as f32;
                    anyhow::ensure!(
                        value.is_finite() && value >= 0.0,
                        "token-cache loss-weight palette must be finite and nonnegative"
                    );
                    Ok(value)
                })
                .collect::<Result<Vec<_>>>()?;
            anyhow::ensure!(
                !palette.is_empty() && palette.len() <= 255,
                "token-cache loss-weight palette must contain 1 to 255 values"
            );
            let run_offsets = read_portable_index_i64(&portable_tensors, "loss_run_offsets")?;
            let run_ends = read_portable_index_i64(&portable_tensors, "loss_run_ends")?;
            let run_codes = read_portable_index_i64(&portable_tensors, "loss_run_codes")?;
            anyhow::ensure!(
                run_offsets.len() == samples + 1 && run_offsets.first() == Some(&0),
                "token-cache loss_run_offsets must contain samples+1 entries starting at zero"
            );
            anyhow::ensure!(
                run_offsets.windows(2).all(|window| window[1] > window[0]),
                "every token-cache sample must contain at least one loss-weight run"
            );
            let run_count = usize::try_from(*run_offsets.last().unwrap())
                .context("token-cache loss-run count exceeds host range")?;
            anyhow::ensure!(
                run_count == run_ends.len() && run_count == run_codes.len(),
                "token-cache loss-run offsets/ends/codes size mismatch"
            );
            anyhow::ensure!(
                run_codes.iter().all(|&code| code >= 0
                    && usize::try_from(code).is_ok_and(|code| code < palette.len())),
                "token-cache loss-run code is outside the declared palette"
            );
            (
                Some(palette),
                Some(run_offsets),
                Some(run_ends),
                Some(run_codes),
            )
        } else {
            (None, None, None, None)
        };

    let data_bytes = fs::read(&data_path)
        .with_context(|| format!("reading token cache {}", data_path.display()))?;
    if let Some(expected_bytes) = success_object
        .get("bytes")
        .and_then(serde_json::Value::as_u64)
    {
        anyhow::ensure!(
            u64::try_from(data_bytes.len()).ok() == Some(expected_bytes),
            "token-cache byte count mismatch: _SUCCESS={expected_bytes}, tokens.bin={}",
            data_bytes.len()
        );
    }
    let expected_tokens_sha = success_object
        .get("tokens_sha256")
        .and_then(serde_json::Value::as_str)
        .context("token-cache _SUCCESS is missing tokens_sha256")?;
    anyhow::ensure!(
        is_sha256_digest(expected_tokens_sha),
        "token-cache tokens_sha256 is malformed"
    );
    let actual_tokens_sha = sha256_hex(&data_bytes);
    anyhow::ensure!(
        actual_tokens_sha.eq_ignore_ascii_case(expected_tokens_sha),
        "token-cache binary checksum failed for {}",
        data_path.display()
    );

    let expected_ordered_sha = success_object
        .get("ordered_record_sha256")
        .and_then(serde_json::Value::as_str)
        .context("token-cache _SUCCESS is missing ordered_record_sha256")?;
    anyhow::ensure!(
        is_sha256_digest(expected_ordered_sha),
        "token-cache ordered_record_sha256 is malformed"
    );
    let ordered_algorithm = success_object
        .get("ordered_record_hash_algorithm")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(TOKEN_CACHE_RECORD_HASH_ALGORITHM);
    anyhow::ensure!(
        ordered_algorithm == TOKEN_CACHE_RECORD_HASH_ALGORITHM,
        "native Vulkan trainer does not understand token-cache record hash algorithm {ordered_algorithm:?}"
    );

    let bytes_per_token = token_bytes
        .checked_add(label_bytes)
        .and_then(|value| value.checked_add(rosa_bytes))
        .context("token-cache record width overflow")?;
    let mut ordered_hasher = Sha256::new();
    ordered_hasher.update(TOKEN_CACHE_RECORD_HASH_HEADER);
    let mut total_tokens = 0u64;
    let mut rows = Vec::with_capacity(samples);
    for row_index in 0..samples {
        let length = usize::try_from(lengths[row_index])
            .context("token-cache row length exceeds host range")?;
        let offset = usize::try_from(offsets[row_index])
            .context("token-cache row offset exceeds host range")?;
        let record_bytes = length
            .checked_mul(bytes_per_token)
            .context("token-cache row byte size overflow")?;
        let end = offset
            .checked_add(record_bytes)
            .context("token-cache row end overflow")?;
        anyhow::ensure!(
            end <= data_bytes.len(),
            "token-cache row {} extends past tokens.bin",
            row_index
        );
        if row_index == 0 {
            anyhow::ensure!(offset == 0, "token-cache first offset must be zero");
        } else {
            let previous_length = usize::try_from(lengths[row_index - 1])?;
            let previous_bytes = previous_length
                .checked_mul(bytes_per_token)
                .context("token-cache previous row byte size overflow")?;
            let expected_offset = usize::try_from(offsets[row_index - 1])?
                .checked_add(previous_bytes)
                .context("token-cache expected offset overflow")?;
            anyhow::ensure!(
                offset == expected_offset,
                "token-cache offsets do not match the declared compact record layout"
            );
        }
        if row_index + 1 == samples {
            anyhow::ensure!(
                end == data_bytes.len(),
                "token-cache data size does not match the final compact record"
            );
        }

        let input_bytes_end = offset + length * token_bytes;
        let input_ids = decode_cache_input_ids(&data_bytes[offset..input_bytes_end], token_dtype)?;
        let labels = if labels_alias_input_ids {
            input_ids.iter().copied().map(i64::from).collect()
        } else {
            let label_end = input_bytes_end + length * label_bytes;
            decode_cache_labels(
                &data_bytes[input_bytes_end..label_end],
                label_dtype.unwrap(),
                label_ignore_sentinel,
            )?
        };
        anyhow::ensure!(
            input_ids.len() == length && labels.len() == length,
            "decoded token-cache record length mismatch"
        );

        let mut expanded_loss_codes = None;
        let loss_weights = if let (
            Some(palette),
            Some(run_offsets),
            Some(run_ends),
            Some(run_codes),
        ) = (
            loss_palette.as_ref(),
            loss_run_offsets.as_ref(),
            loss_run_ends.as_ref(),
            loss_run_codes.as_ref(),
        ) {
            let run_start = usize::try_from(run_offsets[row_index])?;
            let run_stop = usize::try_from(run_offsets[row_index + 1])?;
            anyhow::ensure!(
                run_start < run_stop && run_stop <= run_ends.len(),
                "token-cache loss-run range is invalid for sample {row_index}"
            );
            let mut weights = vec![0.0f32; length];
            let mut codes = vec![0u8; length];
            let mut position = 0usize;
            for run_index in run_start..run_stop {
                let run_end = usize::try_from(run_ends[run_index])?;
                anyhow::ensure!(
                    run_end > position && run_end <= length,
                    "token-cache loss-run ends must increase within each sample and stay inside its length"
                );
                let code = usize::try_from(run_codes[run_index])?;
                weights[position..run_end].fill(palette[code]);
                codes[position..run_end].fill(u8::try_from(code)?);
                position = run_end;
            }
            anyhow::ensure!(
                position == length,
                "token-cache loss runs do not cover sample {row_index}"
            );
            expanded_loss_codes = Some(codes);
            Some(weights)
        } else {
            None
        };

        ordered_hasher.update(u64::try_from(length)?.to_le_bytes());
        ordered_hasher.update(&data_bytes[offset..end]);
        if let Some(codes) = expanded_loss_codes.as_ref() {
            ordered_hasher.update(codes);
        }
        total_tokens = total_tokens
            .checked_add(u64::try_from(length)?)
            .context("token-cache total token count overflow")?;
        rows.push(DatasetRow {
            input_ids,
            labels: Some(labels),
            attention_mask: None,
            loss_weights,
        });
    }
    let ordered_sha = {
        let digest = ordered_hasher.finalize();
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").expect("writing token-cache SHA-256 hex");
        }
        encoded
    };
    anyhow::ensure!(
        ordered_sha.eq_ignore_ascii_case(expected_ordered_sha),
        "token-cache logical record checksum failed: expected={expected_ordered_sha} actual={ordered_sha}"
    );

    let cache_payload = success_object
        .get("cache_payload")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let cache_payload_sha256 = sha256_hex(
        &serde_json::to_vec(&cache_payload).context("serializing token-cache identity payload")?,
    );
    let mut token_cache = serde_json::json!({
        "cache_key": success_object.get("cache_key").and_then(serde_json::Value::as_str).unwrap_or(""),
        "format": success_object.get("format").and_then(serde_json::Value::as_str).unwrap_or(""),
        "ordered_record_sha256": expected_ordered_sha,
        "ordered_record_hash_algorithm": ordered_algorithm,
        "samples": samples,
        "cache_payload_sha256": cache_payload_sha256,
        "audit_sha256": success_object.get("audit_sha256").cloned().unwrap_or(serde_json::Value::Null),
        "tokens_sha256": expected_tokens_sha,
    });
    // Keep the object type explicit so future cache-identity extensions can be
    // inserted without changing the native run-identity schema.
    token_cache
        .as_object_mut()
        .context("native token-cache identity did not encode as an object")?;
    Ok(LoadedDataset {
        rows,
        identity: DatasetSourceIdentity {
            source_kind: "hierarchos-token-cache".to_string(),
            replay_guarantee: "content-addressed-token-cache".to_string(),
            total_tokens,
            token_cache,
        },
    })
}

fn load_training_dataset(path: &Path) -> Result<LoadedDataset> {
    if path.is_dir() {
        return load_token_cache_dataset(path);
    }
    let rows = load_jsonl_dataset_rows(path)?;
    let (ordered_record_sha256, total_tokens) = normalized_dataset_identity(&rows)?;
    let samples = rows.len();
    Ok(LoadedDataset {
        rows,
        identity: DatasetSourceIdentity {
            source_kind: "pretokenized-jsonl".to_string(),
            replay_guarantee: "content-addressed-token-cache".to_string(),
            total_tokens,
            token_cache: serde_json::json!({
                "format": "hierarchos-native-pretokenized-jsonl-v1",
                "ordered_record_sha256": ordered_record_sha256,
                "ordered_record_hash_algorithm": NATIVE_DATASET_HASH_ALGORITHM,
                "samples": samples,
            }),
        },
    })
}

fn stable_native_model_contract(config: &ModelConfig) -> Result<serde_json::Value> {
    let mut contract = serde_json::to_value(config).context("encoding native model contract")?;
    let object = contract
        .as_object_mut()
        .context("native model contract did not encode as an object")?;
    // These fields are learned readiness telemetry updated by checkpoint export,
    // not graph geometry. Binding them would make a checkpoint disagree with
    // the very run identity that produced it.
    for field in [
        "val_proj_alignment_updates",
        "val_proj_alignment_last",
        "val_proj_alignment_ema",
        "val_proj_alignment_best",
        "val_proj_writer_norm",
        "val_proj_trained",
    ] {
        object.remove(field);
    }
    Ok(contract)
}

fn unsigned_run_identity(identity: &serde_json::Value) -> Result<serde_json::Value> {
    let mut unsigned = identity.clone();
    unsigned
        .as_object_mut()
        .context("native run identity must be an object")?
        .remove("sha256");
    Ok(unsigned)
}

fn normalized_unsigned_run_identity(identity: &serde_json::Value) -> Result<serde_json::Value> {
    let unsigned = unsigned_run_identity(identity)?;
    // Normalize through serde_json's parser before hashing. Values created from
    // Rust f32s can otherwise carry a different internal Number representation
    // than the same value after checkpoint JSON has been parsed back in. The
    // normalized Value is also the authoritative semantic comparison form for
    // exact resume, so a disk round-trip cannot manufacture false drift.
    let serialized = serde_json::to_vec(&unsigned).context("serializing native run identity")?;
    serde_json::from_slice(&serialized).context("normalizing native run identity")
}

fn run_identity_digest(identity: &serde_json::Value) -> Result<String> {
    let normalized = normalized_unsigned_run_identity(identity)?;
    Ok(sha256_hex(
        &serde_json::to_vec(&normalized).context("serializing normalized native run identity")?,
    ))
}

fn native_run_identity(
    args: &Args,
    dataset: &[DatasetRow],
    dataset_identity: &DatasetSourceIdentity,
    config: &ModelConfig,
    full_batch_count: usize,
) -> Result<serde_json::Value> {
    let model_contract = stable_native_model_contract(config)?;
    let model_contract_sha256 = sha256_hex(
        &serde_json::to_vec(&model_contract).context("serializing native model contract")?,
    );
    let mut objective = effective_training_config(args);
    let objective = objective
        .as_object_mut()
        .context("native effective training config did not encode as an object")?;
    // Target duration and hardware/scheduling choices can change without
    // changing the next mathematical update. All optimizer, loss, recurrence,
    // sampler, and LR-curve fields remain identity-bound.
    for field in [
        "backend",
        "epochs",
        "device_index",
        "device_indices",
        "gradient_stream_chunk_values",
    ] {
        objective.remove(field);
    }

    let mut identity = serde_json::json!({
        "format": NATIVE_RUN_IDENTITY_FORMAT,
        "version": 1,
        "objective": serde_json::Value::Object(objective.clone()),
        "dataset": {
            "source_kind": dataset_identity.source_kind,
            "replay_guarantee": dataset_identity.replay_guarantee,
            "row_count": dataset.len(),
            "total_tokens": dataset_identity.total_tokens,
        },
        "token_cache": dataset_identity.token_cache.clone(),
        "loader": {
            "format": "hierarchos-portable-native-loader-v1",
            "dataset_len": dataset.len(),
            "dataloader_len": full_batch_count,
            "drop_last": true,
        },
        "model": {
            "architecture_revision": config.architecture_revision,
            "architecture_contract_sha256": config.architecture_contract_sha256,
            "native_model_contract_sha256": model_contract_sha256,
        },
        "optimizer_grouping_version": 2,
    });
    let digest = run_identity_digest(&identity)?;
    identity
        .as_object_mut()
        .context("native run identity did not encode as an object")?
        .insert("sha256".to_string(), serde_json::Value::String(digest));
    Ok(identity)
}

fn validate_native_run_identity_value(
    saved: &serde_json::Value,
    current: &serde_json::Value,
) -> Result<()> {
    let saved_format = saved
        .get("format")
        .and_then(serde_json::Value::as_str)
        .context("saved run_identity is missing its format")?;
    anyhow::ensure!(
        saved_format == NATIVE_RUN_IDENTITY_FORMAT,
        "native exact resume does not yet understand run identity format {saved_format:?}; use a native v1 checkpoint or a deliberate weights-only --model continuation"
    );
    let saved_digest = saved
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .context("saved native run_identity is missing sha256")?;
    anyhow::ensure!(
        saved_digest.len() == 64 && saved_digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "saved native run_identity has a malformed sha256"
    );
    let saved_unsigned = normalized_unsigned_run_identity(saved)?;
    let current_unsigned = normalized_unsigned_run_identity(current)?;
    let recomputed = run_identity_digest(saved)?;
    // The earliest native-v1 writer hashed serde_json Values before a
    // serialize/parse normalization pass. A float-valued identity could then
    // reload with bytewise-equivalent JSON semantics but a different self hash.
    // Exact-resume safety is therefore based on the full unsigned identity, as
    // it is in the PyTorch trainer; the digest remains a content-addressed
    // fingerprint and malformed digest fields still fail closed above.
    if saved_unsigned != current_unsigned {
        let saved_data = saved
            .pointer("/token_cache/ordered_record_sha256")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<missing>");
        let current_data = current
            .pointer("/token_cache/ordered_record_sha256")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<missing>");
        let saved_identity = saved_digest;
        let current_identity = current
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<missing>");
        let self_digest_status = if saved_digest.eq_ignore_ascii_case(&recomputed) {
            "valid"
        } else {
            "legacy-or-corrupt"
        };
        bail!(
            "exact native resume identity mismatch: saved={saved_identity} current={current_identity} saved_self_digest={self_digest_status} dataset_saved={saved_data} dataset_current={current_data}; refusing to apply optimizer/cursor state to a different data objective"
        );
    }
    Ok(())
}

fn validate_native_run_identity_package(
    package_dir: &Path,
    current: &serde_json::Value,
) -> Result<()> {
    let manifest_path = package_dir.join(HIERARCHOS_VULKAN_TRAINING_MANIFEST_FILENAME);
    let manifest: HierarchosTrainingCheckpointManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?,
    )
    .with_context(|| format!("decoding {}", manifest_path.display()))?;
    let replay_file = manifest
        .portable_replay_file
        .as_deref()
        .context("exact native resume manifest is missing portable_replay_file")?;
    anyhow::ensure!(
        Path::new(replay_file).components().count() == 1,
        "exact native resume portable_replay_file must be package-local"
    );
    let saved = read_portable_replay_json_field(&package_dir.join(replay_file), "run_identity")?
        .context("exact native resume replay is missing run_identity")?;
    validate_native_run_identity_value(&saved, current)
}

fn pack_batch(rows: &[&DatasetRow], vocab_size: usize) -> Result<PackedBatch> {
    let batch = rows.len();
    anyhow::ensure!(batch > 0, "cannot pack an empty batch");
    let tokens = rows
        .iter()
        .map(|row| row.input_ids.len())
        .max()
        .context("cannot determine batch token width")?;
    let len = batch
        .checked_mul(tokens)
        .context("packed batch shape overflow")?;
    let mut input_ids = vec![0u32; len];
    let mut labels = vec![-100i64; len];
    let mut attention_mask = vec![0.0f32; len];
    let mut loss_weights = vec![0.0f32; len];

    for (row_index, row) in rows.iter().enumerate() {
        let labels_row = row
            .labels
            .as_ref()
            .context("validated row is missing labels")?;
        for token_index in 0..row.input_ids.len() {
            let token_id = row.input_ids[token_index];
            anyhow::ensure!(
                (token_id as usize) < vocab_size,
                "token id {token_id} in batch row {row_index} exceeds vocabulary size {vocab_size}"
            );
            let offset = row_index * tokens + token_index;
            input_ids[offset] = token_id;
            labels[offset] = labels_row[token_index];
            attention_mask[offset] = row
                .attention_mask
                .as_ref()
                .map_or(1.0, |mask| mask[token_index]);
            loss_weights[offset] = row
                .loss_weights
                .as_ref()
                .map_or(1.0, |weights| weights[token_index]);
        }
    }
    Ok(PackedBatch {
        tokens,
        input_ids,
        labels,
        attention_mask,
        loss_weights,
    })
}

fn update_mode(
    batch_index: usize,
    batch_count: usize,
    gradient_accumulation_steps: usize,
) -> HierarchosTokenTapeUpdateMode {
    let group_offset = batch_index % gradient_accumulation_steps;
    let group_start = group_offset == 0;
    let group_end =
        group_offset + 1 == gradient_accumulation_steps || batch_index + 1 == batch_count;
    match (group_start, group_end) {
        (true, true) => HierarchosTokenTapeUpdateMode::Step,
        (true, false) => HierarchosTokenTapeUpdateMode::BeginAccumulation,
        (false, true) => HierarchosTokenTapeUpdateMode::FinishAccumulation,
        (false, false) => HierarchosTokenTapeUpdateMode::Accumulate,
    }
}

fn dynamic_accumulation_mode(
    batch_index: usize,
    gradient_accumulation_steps: usize,
) -> HierarchosTokenTapeUpdateMode {
    if batch_index % gradient_accumulation_steps == 0 {
        HierarchosTokenTapeUpdateMode::BeginAccumulation
    } else {
        HierarchosTokenTapeUpdateMode::Accumulate
    }
}

fn optimizer_window_ends(
    batch_index: usize,
    batch_count: usize,
    gradient_accumulation_steps: usize,
) -> bool {
    batch_index % gradient_accumulation_steps + 1 == gradient_accumulation_steps
        || batch_index + 1 == batch_count
}

fn precision_uses_dynamic_loss_scaling(policy: HierarchosTrainingPrecisionPolicy) -> bool {
    matches!(
        policy,
        HierarchosTrainingPrecisionPolicy::Fp16StorageParity
            | HierarchosTrainingPrecisionPolicy::Fp16StorageFp16LmBackward
    )
}

fn fresh_pytorch_grad_scaler_state(initial_scale: f64) -> HierarchosLossScalingState {
    // Match torch.amp.GradScaler defaults so a Vulkan checkpoint can resume on
    // PyTorch/CUDA without a backend-specific scaler translation. The scale is
    // configurable because a hardware-qualified Vulkan target may prefer to
    // begin below PyTorch's generic 65536 default and avoid deterministic
    // overflow/backoff windows during native half training.
    HierarchosLossScalingState {
        mode: "dynamic".to_string(),
        scale: Some(initial_scale),
        growth_factor: Some(2.0),
        backoff_factor: Some(0.5),
        growth_interval: Some(2_000),
        growth_tracker: Some(0),
        pending_gradients_scaled: false,
    }
}

fn training_execution_policy(
    precision: HierarchosTrainingPrecisionPolicy,
    loss_scaling: Option<&HierarchosLossScalingState>,
) -> HierarchosExecutionPolicyState {
    let amp_enabled = precision_uses_dynamic_loss_scaling(precision);
    HierarchosExecutionPolicyState {
        format: HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT.to_string(),
        source_backend: "vulkan".to_string(),
        compute_dtype: if amp_enabled { "float16" } else { "float32" }.to_string(),
        autocast_enabled: amp_enabled,
        // The native full-model trainer has no stochastic model op today. Keep
        // the portable declaration explicit so PyTorch resume does not demand
        // an unrelated backend RNG blob.
        stochastic_rng: HierarchosStochasticRngPolicyState {
            mode: "none".to_string(),
            state_required: false,
            canonical_counter: None,
        },
        loss_scaling: loss_scaling.cloned().unwrap_or(HierarchosLossScalingState {
            mode: "none".to_string(),
            scale: None,
            growth_factor: None,
            backoff_factor: None,
            growth_interval: None,
            growth_tracker: None,
            pending_gradients_scaled: false,
        }),
    }
}

fn contiguous_weighted_shard_ranges(
    weights: &[usize],
    max_shards: usize,
) -> Result<Vec<Range<usize>>> {
    anyhow::ensure!(max_shards > 0, "multi-device shard count must be positive");
    let equal_capacities = vec![1.0f64; max_shards];
    contiguous_weighted_shard_ranges_by_capacity(weights, &equal_capacities)
}

fn contiguous_weighted_shard_ranges_by_capacity(
    weights: &[usize],
    lane_capacities: &[f64],
) -> Result<Vec<Range<usize>>> {
    anyhow::ensure!(
        !weights.is_empty(),
        "cannot shard an empty optimizer window"
    );
    anyhow::ensure!(
        !lane_capacities.is_empty(),
        "multi-device lane capacity vector must not be empty"
    );
    anyhow::ensure!(
        weights.iter().all(|&weight| weight > 0),
        "multi-device shard weights must be positive"
    );
    anyhow::ensure!(
        lane_capacities
            .iter()
            .all(|capacity| capacity.is_finite() && *capacity > 0.0),
        "multi-device lane capacities must be finite and positive"
    );
    let shard_count = weights.len().min(lane_capacities.len());
    let total_weight = weights.iter().try_fold(0usize, |total, &weight| {
        total
            .checked_add(weight)
            .context("multi-device shard weight overflow")
    })?;
    let mut start = 0usize;
    let mut consumed_weight = 0usize;
    let mut ranges = Vec::with_capacity(shard_count);
    for shard_index in 0..shard_count {
        let shards_left = shard_count - shard_index;
        let max_end = weights.len() - (shards_left - 1);
        let end = if shards_left == 1 {
            weights.len()
        } else {
            let remaining_weight = total_weight
                .checked_sub(consumed_weight)
                .context("multi-device consumed shard weight exceeded total")?;
            let remaining_capacity = lane_capacities[shard_index..shard_count]
                .iter()
                .sum::<f64>();
            anyhow::ensure!(
                remaining_capacity.is_finite() && remaining_capacity > 0.0,
                "multi-device remaining lane capacity is invalid"
            );
            let target_weight =
                remaining_weight as f64 * lane_capacities[shard_index] / remaining_capacity;
            let mut candidate_end = start + 1;
            let mut candidate_weight = weights[start];
            while candidate_end < max_end {
                let next_weight = candidate_weight
                    .checked_add(weights[candidate_end])
                    .context("multi-device shard weight overflow")?;
                let candidate_distance = (candidate_weight as f64 - target_weight).abs();
                let next_distance = (next_weight as f64 - target_weight).abs();
                if next_distance >= candidate_distance {
                    break;
                }
                candidate_weight = next_weight;
                candidate_end += 1;
            }
            candidate_end
        };
        let shard_weight = weights[start..end]
            .iter()
            .try_fold(0usize, |total, &weight| {
                total
                    .checked_add(weight)
                    .context("multi-device shard weight overflow")
            })?;
        consumed_weight = consumed_weight
            .checked_add(shard_weight)
            .context("multi-device consumed shard weight overflow")?;
        ranges.push(start..end);
        start = end;
    }
    anyhow::ensure!(
        start == weights.len(),
        "multi-device shard partition lost work"
    );
    anyhow::ensure!(
        consumed_weight == total_weight,
        "multi-device shard partition lost token weight"
    );
    Ok(ranges)
}

fn update_steps_per_epoch(batch_count: usize, accumulation_steps: usize) -> Result<u64> {
    let updates = batch_count
        .checked_add(accumulation_steps - 1)
        .context("optimizer update-count overflow")?
        / accumulation_steps;
    u64::try_from(updates).context("optimizer update-count exceeds u64 range")
}

fn resolve_warmup_steps(explicit_steps: u64, warmup_ratio: f64, total_steps: u64) -> u64 {
    let max_warmup = total_steps.saturating_sub(1);
    if explicit_steps > 0 {
        return explicit_steps.min(max_warmup);
    }
    if warmup_ratio <= 0.0 {
        return 0;
    }
    ((total_steps as f64 * warmup_ratio.min(1.0)).ceil() as u64).min(max_warmup)
}

fn scheduled_lr_at_step(
    max_lr: f64,
    min_lr: f64,
    resolved_warmup_steps: u64,
    total_steps: u64,
    step: u64,
) -> f64 {
    let step = step.min(total_steps);
    if resolved_warmup_steps > 0 && step < resolved_warmup_steps {
        let progress = (step + 1) as f64 / resolved_warmup_steps as f64;
        return min_lr + (max_lr - min_lr) * progress.min(1.0);
    }
    let decay_steps = (total_steps - resolved_warmup_steps).max(1);
    let decay_step = step.saturating_sub(resolved_warmup_steps).min(decay_steps);
    let cosine =
        0.5 * (1.0 + (std::f64::consts::PI * decay_step as f64 / decay_steps as f64).cos());
    min_lr + (max_lr - min_lr) * cosine
}

fn new_lr_schedule(
    args: &Args,
    total_steps: u64,
    optimizer_step: u64,
) -> HierarchosLearningRateScheduleState {
    if args.disable_lr_schedule {
        return HierarchosLearningRateScheduleState {
            enabled: false,
            step: None,
            total_steps: None,
            max_lr: None,
            min_lr: None,
            warmup_steps: None,
            warmup_ratio: None,
            resolved_warmup_steps: None,
            base_lrs: Vec::new(),
            last_lrs: Vec::new(),
            step_count: None,
        };
    }

    let total_steps = total_steps.max(1);
    let step = optimizer_step.min(total_steps);
    let max_lr = f64::from(args.learning_rate);
    let min_lr = f64::from(args.min_learning_rate);
    let resolved_warmup_steps =
        resolve_warmup_steps(args.warmup_steps, args.warmup_ratio, total_steps);
    let live_lr = scheduled_lr_at_step(max_lr, min_lr, resolved_warmup_steps, total_steps, step);
    HierarchosLearningRateScheduleState {
        enabled: true,
        step: Some(step),
        total_steps: Some(total_steps),
        max_lr: Some(max_lr),
        min_lr: Some(min_lr),
        warmup_steps: Some(args.warmup_steps.min(total_steps)),
        warmup_ratio: Some(args.warmup_ratio),
        resolved_warmup_steps: Some(resolved_warmup_steps),
        // PyTorch's canonical Hierarchos AdamW always has decay/no-decay groups.
        base_lrs: vec![max_lr, max_lr],
        last_lrs: vec![live_lr, live_lr],
        step_count: Some(step.saturating_add(1)),
    }
}

fn schedule_live_lr(
    schedule: &HierarchosLearningRateScheduleState,
    fallback_lr: f32,
) -> Result<f32> {
    if !schedule.enabled {
        return Ok(fallback_lr);
    }
    anyhow::ensure!(
        schedule.last_lrs.len() == 2,
        "native main LR scheduler must carry the two canonical AdamW group LRs"
    );
    let first = schedule.last_lrs[0];
    anyhow::ensure!(
        schedule.last_lrs.iter().all(|&lr| lr == first),
        "Vulkan currently requires identical live LRs for the canonical decay/no-decay groups"
    );
    anyhow::ensure!(
        first.is_finite() && first >= 0.0 && first <= f64::from(f32::MAX),
        "native main LR scheduler contains an invalid live LR {first}"
    );
    Ok(first as f32)
}

fn advance_lr_schedule(schedule: &mut HierarchosLearningRateScheduleState) -> Result<()> {
    if !schedule.enabled {
        return Ok(());
    }
    let total_steps = schedule
        .total_steps
        .context("enabled native LR scheduler is missing total_steps")?;
    let step = schedule
        .step
        .context("enabled native LR scheduler is missing step")?
        .saturating_add(1);
    let max_lr = schedule
        .max_lr
        .context("enabled native LR scheduler is missing max_lr")?;
    let min_lr = schedule.min_lr.unwrap_or(0.0);
    let resolved_warmup_steps = schedule.resolved_warmup_steps.unwrap_or(0);
    let live_lr = scheduled_lr_at_step(max_lr, min_lr, resolved_warmup_steps, total_steps, step);
    schedule.step = Some(step);
    schedule.last_lrs = vec![live_lr, live_lr];
    schedule.step_count = Some(step.saturating_add(1));
    Ok(())
}

fn new_data_cursor(args: &Args, dataset_size: usize) -> Result<HierarchosDataStreamCursorState> {
    Ok(HierarchosDataStreamCursorState {
        format: HIERARCHOS_VULKAN_DATA_STREAM_CURSOR_FORMAT.to_string(),
        sampler_kind: "epoch-shuffle".to_string(),
        rng_algorithm: HIERARCHOS_VULKAN_PORTABLE_SAMPLER_RNG_ALGORITHM.to_string(),
        seed: args.seed,
        epoch: 0,
        batch_cursor: 0,
        dataset_size: u64::try_from(dataset_size).context("dataset size exceeds u64 range")?,
        batch_size: u64::try_from(args.batch_size).context("batch size exceeds u64 range")?,
        shuffle: args.shuffle,
        drop_last: true,
        bucket_size: None,
        preserve_order: false,
    })
}

fn validate_resume_cursor(
    cursor: &HierarchosDataStreamCursorState,
    args: &Args,
    dataset_size: usize,
) -> Result<()> {
    match cursor.sampler_kind.as_str() {
        "epoch-shuffle" => {
            anyhow::ensure!(
                cursor.bucket_size.is_none() && !cursor.preserve_order,
                "epoch-shuffle resume cursor cannot carry length-bucket policy"
            );
        }
        "length-grouped-batch" => {
            let bucket_size = cursor
                .bucket_size
                .context("length-grouped resume cursor is missing bucket_size")?;
            anyhow::ensure!(
                bucket_size >= cursor.batch_size && bucket_size % cursor.batch_size == 0,
                "length-grouped resume bucket_size {} must be a positive multiple of batch_size {}",
                bucket_size,
                cursor.batch_size
            );
        }
        other => bail!("native trainer resume does not support sampler kind {other:?}"),
    }
    anyhow::ensure!(
        cursor.dataset_size == dataset_size as u64,
        "resume dataset size {} does not match current dataset size {}",
        cursor.dataset_size,
        dataset_size
    );
    anyhow::ensure!(
        cursor.batch_size == args.batch_size as u64,
        "resume batch size {} does not match --batch-size {}",
        cursor.batch_size,
        args.batch_size
    );
    anyhow::ensure!(
        cursor.seed == args.seed,
        "resume sampler seed {} does not match --seed {}",
        cursor.seed,
        args.seed
    );
    anyhow::ensure!(
        cursor.shuffle == args.shuffle,
        "resume shuffle policy does not match current CLI flags"
    );
    anyhow::ensure!(
        cursor.drop_last,
        "native trainer resume requires drop_last=true"
    );
    Ok(())
}

fn effective_training_config(args: &Args) -> serde_json::Value {
    serde_json::json!({
        "backend": "vulkan",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "persist_state": args.persist_state,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "device_index": args.device_index,
        "device_indices": args.device_indices,
        "gradient_stream_chunk_values": args.gradient_stream_chunk_values,
        "starting_lr": args.learning_rate,
        "min_lr": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "disable_lr_schedule": args.disable_lr_schedule,
        "tbptt_chunk_size": args.tbptt_chunk_size,
        "trainable_prefixes": args.trainable_prefixes,
        "grad_clip": args.grad_clip,
        "initial_loss_scale": args.initial_loss_scale,
        "z_loss_weight": args.z_loss_weight,
        "ponder_loss_weight": args.ponder_loss_weight,
        "commitment_loss_weight": args.commitment_loss_weight,
        "max_ce_loss_for_backward": args.max_ce_loss_for_backward,
        "max_ponder_cost_for_backward": args.max_ponder_cost_for_backward,
        "max_commitment_cost_for_backward": args.max_commitment_cost_for_backward,
        "max_skipped_train_batches": args.max_skipped_train_batches,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    })
}

fn validate_resume_f32_config(
    config: &serde_json::Value,
    key: &str,
    current: f32,
    cli_name: &str,
) -> Result<()> {
    let Some(saved) = config.get(key) else {
        return Ok(());
    };
    let saved = saved
        .as_f64()
        .with_context(|| format!("resume training config field {key:?} is not numeric"))?;
    anyhow::ensure!(
        saved.is_finite(),
        "resume training config field {key:?} must be finite"
    );
    let saved_f32 = saved as f32;
    anyhow::ensure!(
        saved_f32.to_bits() == current.to_bits(),
        "resume {key}={saved_f32} does not match {cli_name} {current}; exact continuation forbids changing optimizer/objective safety policy"
    );
    Ok(())
}

fn validate_resume_numerical_policy(config: &serde_json::Value, args: &Args) -> Result<()> {
    for (key, current, cli_name) in [
        ("grad_clip", args.grad_clip, "--grad-clip"),
        ("z_loss_weight", args.z_loss_weight, "--z-loss-weight"),
        (
            "ponder_loss_weight",
            args.ponder_loss_weight,
            "--ponder-loss-weight",
        ),
        (
            "commitment_loss_weight",
            args.commitment_loss_weight,
            "--commitment-loss-weight",
        ),
        (
            "max_ce_loss_for_backward",
            args.max_ce_loss_for_backward,
            "--max-ce-loss-for-backward",
        ),
        (
            "max_ponder_cost_for_backward",
            args.max_ponder_cost_for_backward,
            "--max-ponder-cost-for-backward",
        ),
        (
            "max_commitment_cost_for_backward",
            args.max_commitment_cost_for_backward,
            "--max-commitment-cost-for-backward",
        ),
        ("beta1", args.beta1, "--beta1"),
        ("beta2", args.beta2, "--beta2"),
        ("eps", args.eps, "--eps"),
        ("weight_decay", args.weight_decay, "--weight-decay"),
    ] {
        validate_resume_f32_config(config, key, current, cli_name)?;
    }
    Ok(())
}

fn account_skipped_training_batch(
    skipped_train_batches: &mut u64,
    max_skipped_train_batches: u64,
    epoch: u64,
    batch_index: usize,
    reason: &str,
) -> Result<()> {
    *skipped_train_batches = skipped_train_batches
        .checked_add(1)
        .context("native skipped-train-batch counter overflow")?;
    if *skipped_train_batches > max_skipped_train_batches {
        bail!(
            "Training skip/error budget exceeded at epoch={}, batch={}: {}. Observed {}, allowed {}.",
            epoch + 1,
            batch_index + 1,
            reason,
            *skipped_train_batches,
            max_skipped_train_batches,
        );
    }
    eprintln!(
        "WARNING: Training batch skipped within explicit budget ({}/{}): {} (epoch={}, batch={}).",
        *skipped_train_batches,
        max_skipped_train_batches,
        reason,
        epoch + 1,
        batch_index + 1,
    );
    Ok(())
}

fn training_session(
    args: &Args,
    cursor: &HierarchosDataStreamCursorState,
    schedule: &HierarchosLearningRateScheduleState,
    skipped_train_batches: u64,
    execution_policy: HierarchosExecutionPolicyState,
) -> HierarchosTrainingSessionState {
    HierarchosTrainingSessionState {
        format: HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT.to_string(),
        completed_epoch: cursor.epoch,
        mid_epoch_step: cursor.batch_cursor,
        optimizer_grouping_version: 2,
        main_lr_scheduler: Some(schedule.clone()),
        ltm_lr_scheduler: None,
        effective_training_config: effective_training_config(args),
        skipped_train_batches,
        data_stream_cursor: Some(cursor.clone()),
        execution_policy: Some(execution_policy),
    }
}

fn assemble_portable_replay_state(
    run_identity: &serde_json::Value,
    running_carriers: Option<&HierarchosPortableRunningCarriers>,
) -> Result<(serde_json::Value, Vec<HierarchosPortableReplayTensor>)> {
    if !run_identity.is_object() {
        bail!("native run identity must be a JSON object");
    }
    let mut items = vec![serde_json::json!([
        "run_identity",
        encode_portable_replay_json(run_identity)?
    ])];
    let mut tensors = Vec::new();
    if let Some(carriers) = running_carriers {
        let (encoded, carrier_tensors) = encode_portable_running_carriers(carriers)?;
        items.push(serde_json::json!(["running_states", encoded]));
        tensors.extend(carrier_tensors);
    }
    Ok((
        serde_json::json!({"__kind__": "dict", "items": items}),
        tensors,
    ))
}

#[allow(clippy::too_many_arguments)]
fn portable_training_replay(
    graph: &HierarchosTrainingGraph,
    args: &Args,
    cursor: &HierarchosDataStreamCursorState,
    schedule: &HierarchosLearningRateScheduleState,
    skipped_train_batches: u64,
    execution_policy: HierarchosExecutionPolicyState,
    carried_h_state: Option<&[f32]>,
    carried_l_state: Option<&[f32]>,
    previous_context: &[f32],
    target_context: &[f32],
    ltm_state: Option<&HierarchosPortableLtmRunningState>,
    run_identity: &serde_json::Value,
) -> Result<HierarchosPortableTrainingReplay> {
    let session = training_session(
        args,
        cursor,
        schedule,
        skipped_train_batches,
        execution_policy,
    );
    let running_carriers = if args.persist_state && cursor.batch_cursor > 0 {
        let h_state = carried_h_state
            .context("persisted mid-epoch checkpoint is missing the carried H state")?;
        let l_state = carried_l_state
            .context("persisted mid-epoch checkpoint is missing the carried L state")?;
        let ltm_state = ltm_state
            .context("persisted mid-epoch checkpoint is missing portable LTM metadata")?
            .clone();
        Some(
            graph
                .snapshot_portable_running_carriers(
                    args.batch_size,
                    h_state,
                    l_state,
                    previous_context,
                    target_context,
                    ltm_state,
                )
                .context("snapshotting PyTorch-compatible Vulkan running_states")?,
        )
    } else {
        None
    };
    // Assemble the complete fail-closed host state before invoking the replay
    // constructor. Validation deliberately requires run_identity (and, for a
    // persisted stream, running_states) at every mid-epoch cursor.
    let (encoded_state, tensors) =
        assemble_portable_replay_state(run_identity, running_carriers.as_ref())?;
    HierarchosPortableTrainingReplay::new_with_training_session(
        cursor.epoch,
        cursor.batch_cursor,
        session,
        encoded_state,
        tensors,
    )
}

fn periodic_checkpoint_due(batch_number: usize, save_steps: u64) -> bool {
    save_steps > 0 && (batch_number as u64) % save_steps == 0
}

fn periodic_checkpoint_crossed(
    completed_before: usize,
    completed_after: usize,
    save_steps: u64,
) -> bool {
    save_steps > 0 && (completed_before as u64) / save_steps < (completed_after as u64) / save_steps
}

fn periodic_checkpoint_dir(output_dir: &Path, epoch: u64, batch_number: usize) -> PathBuf {
    output_dir.join(format!(
        "checkpoint-epoch-{}-step-{}",
        epoch + 1,
        batch_number
    ))
}

#[derive(Debug)]
struct BudgetedWindowTrainSummary {
    optimizer_step: u32,
    queue_submissions: u64,
    optimizer_wavefront_windows: usize,
    optimizer_wavefront_ranges: usize,
    losses: Vec<f32>,
    gradient_reductions: usize,
    replica_broadcasts: usize,
    replica_broadcast_compute_handoffs: usize,
    gradient_stream_chunks: usize,
    gradient_stream_values: usize,
    gradient_stream_pipeline_slots: usize,
    gradient_stream_persistent_reuses: usize,
    gradient_stream_peak_host_bytes: usize,
    gradient_stream_peak_device_bytes: usize,
    gradient_stream_peak_host_heap_bytes: usize,
    gradient_stream_backends: HashSet<VulkanGradientTransportBackend>,
    replica_state_stream_chunks: usize,
    replica_state_stream_values: usize,
    replica_state_stream_pipeline_slots: usize,
    replica_state_stream_persistent_reuses: usize,
    replica_state_stream_peak_host_bytes: usize,
    replica_state_stream_peak_device_bytes: usize,
    replica_state_stream_backends: HashSet<VulkanGradientTransportBackend>,
    replica_state_host_fallbacks: usize,
    runtime_device_profiles: Vec<JointRuntimeDeviceWindowProfile>,
    runtime_phase_profile: JointRuntimePhaseWindowProfile,
}

#[derive(Default)]
struct ReplicaStateBroadcastSummary {
    queue_submissions: u64,
    stream_chunks: usize,
    stream_values: usize,
    stream_pipeline_slots: usize,
    stream_persistent_reuses: usize,
    stream_peak_host_bytes: usize,
    stream_peak_device_bytes: usize,
    stream_backends: HashSet<VulkanGradientTransportBackend>,
    host_fallbacks: usize,
}

struct OwnedRawTokenLabeledSequenceInput {
    tokens: usize,
    input_ids: Vec<u32>,
    labels: Vec<i64>,
    attention_mask: Option<Vec<f32>>,
    loss_weights: Option<Vec<f32>>,
    initial_previous_context: Vec<f32>,
    initial_target_context: Vec<f32>,
    global_pos_offset: u64,
    reset_rosa_at_start: bool,
    pytorch_tbptt_chunk_size: Option<usize>,
}

impl OwnedRawTokenLabeledSequenceInput {
    fn capture(input: &HierarchosRawTokenLabeledSequenceInput<'_>) -> Self {
        Self {
            tokens: input.tokens,
            input_ids: input.input_ids.to_vec(),
            labels: input.labels.to_vec(),
            attention_mask: input.attention_mask.map(<[f32]>::to_vec),
            loss_weights: input.loss_weights.map(<[f32]>::to_vec),
            initial_previous_context: input.initial_previous_context.to_vec(),
            initial_target_context: input.initial_target_context.to_vec(),
            global_pos_offset: input.global_pos_offset,
            reset_rosa_at_start: input.reset_rosa_at_start,
            pytorch_tbptt_chunk_size: input.pytorch_tbptt_chunk_size,
        }
    }

    fn as_borrowed(&self) -> HierarchosRawTokenLabeledSequenceInput<'_> {
        HierarchosRawTokenLabeledSequenceInput {
            tokens: self.tokens,
            input_ids: &self.input_ids,
            labels: &self.labels,
            attention_mask: self.attention_mask.as_deref(),
            loss_weights: self.loss_weights.as_deref(),
            initial_previous_context: &self.initial_previous_context,
            initial_target_context: &self.initial_target_context,
            global_pos_offset: self.global_pos_offset,
            reset_rosa_at_start: self.reset_rosa_at_start,
            pytorch_tbptt_chunk_size: self.pytorch_tbptt_chunk_size,
        }
    }
}

struct ReplicaStateBroadcastJob {
    replica_index: usize,
    source: Arc<HierarchosFullModelReplicaTransportSource>,
    reset_accumulation: bool,
    chunk_values: usize,
    range_retirement: Option<HierarchosReplicaStateRangeRetirement>,
    timeline_reservation: Option<HierarchosReplicaStateTimelineReservation>,
    portable_state:
        Arc<OnceLock<Result<Arc<hierarchos_vulkan::HierarchosFullModelReplicaState>, String>>>,
    completion: mpsc::Sender<ReplicaStateBroadcastCompletion>,
}

struct ReplicaStateBroadcastCompletion {
    replica_index: usize,
    result: Result<ReplicaStateBroadcastSummary>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct PhaseAwareTapeGeometry {
    sequence_microbatch_size: usize,
    state_checkpoint_stride: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct JointRuntimeScheduleArm {
    gradient_stream_chunk_values: usize,
    tape_geometry: PhaseAwareTapeGeometry,
    optimizer_broadcast_overlap: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct SubmissionLatencyDelta {
    samples: u64,
    total_ns: u64,
    kernel_profile_samples: u64,
    kernel_dispatches: u64,
    kernel_gpu_ns_total: u64,
}

impl SubmissionLatencyDelta {
    fn between(before: VulkanSubmissionArenaStats, after: VulkanSubmissionArenaStats) -> Self {
        Self {
            samples: after
                .timeline_retirement_latency_samples
                .saturating_sub(before.timeline_retirement_latency_samples),
            total_ns: after
                .timeline_retirement_latency_ns_total
                .saturating_sub(before.timeline_retirement_latency_ns_total),
            kernel_profile_samples: after
                .kernel_timestamp_profile_samples
                .saturating_sub(before.kernel_timestamp_profile_samples),
            kernel_dispatches: after
                .kernel_timestamp_profile_dispatches
                .saturating_sub(before.kernel_timestamp_profile_dispatches),
            kernel_gpu_ns_total: after
                .kernel_timestamp_profile_gpu_ns_total
                .saturating_sub(before.kernel_timestamp_profile_gpu_ns_total),
        }
    }

    fn average_ns(self) -> Option<f64> {
        (self.samples != 0).then(|| self.total_ns as f64 / self.samples as f64)
    }
}

#[derive(Clone, Debug, Default)]
struct JointRuntimeDeviceWindowProfile {
    lane_index: usize,
    tokens: u64,
    elapsed_seconds: f64,
    queue_submissions: u64,
    latency: SubmissionLatencyDelta,
    shared_submission_arena: bool,
    device_local_usage_ratio: Option<f64>,
    device_local_pressure_bucket: Option<u8>,
}

impl JointRuntimeDeviceWindowProfile {
    fn tokens_per_second(&self) -> Option<f64> {
        (self.tokens != 0 && self.elapsed_seconds.is_finite() && self.elapsed_seconds > 0.0)
            .then(|| self.tokens as f64 / self.elapsed_seconds)
    }
}

fn joint_runtime_memory_pressure(
    graph: &HierarchosTrainingGraph,
    collect_runtime_profile: bool,
) -> Result<(Option<f64>, Option<u8>)> {
    if !collect_runtime_profile {
        return Ok((None, None));
    }
    let budget = graph
        .memory_budget()
        .context("querying Vulkan memory pressure for joint-runtime telemetry")?;
    let usage_ratio = (budget.device_local_budget_bytes != 0)
        .then(|| budget.device_local_usage_bytes as f64 / budget.device_local_budget_bytes as f64);
    Ok((usage_ratio, budget.device_local_pressure_bucket()))
}

/// Host-observed service time for scheduler-controlled phases inside one clean
/// optimizer window. These are intentionally narrower than whole-window wall
/// time: the gradient value is the sum of synchronous replica-gradient stream
/// reduction calls, while the optimizer value covers the AdamW / preceding
/// broadcast-retirement boundary. That makes each signal useful to a future
/// phase-specific selector without changing the current global arm policy.
#[derive(Clone, Copy, Debug, Default)]
struct JointRuntimePhaseWindowProfile {
    gradient_reduction_service_seconds: f64,
    optimizer_boundary_service_seconds: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct JointRuntimeDeviceMeasurements {
    lane_index: usize,
    windows: u64,
    tokens: u64,
    elapsed_seconds: f64,
    queue_submissions: u64,
    timeline_latency_samples: u64,
    timeline_latency_ns_total: u64,
    kernel_profile_samples: u64,
    kernel_dispatches: u64,
    kernel_gpu_ns_total: u64,
    shared_submission_arena_windows: u64,
    adaptive_tokens_per_second: Option<f64>,
    /// Per-lane steady-state throughput moments. These are persisted separately
    /// from the EWMA so heterogeneous-device sharding can distinguish a truly
    /// faster adapter from a noisy/thermally unstable one. `serde(default)`
    /// keeps profiles written before lane-confidence tracking readable.
    #[serde(default)]
    confidence_throughput_samples: u64,
    #[serde(default)]
    confidence_throughput_mean: Option<f64>,
    #[serde(default)]
    confidence_throughput_m2: f64,
    #[serde(default)]
    adaptive_device_local_usage_ratio: Option<f64>,
    #[serde(default)]
    peak_device_local_usage_ratio: Option<f64>,
    #[serde(default)]
    max_device_local_pressure_bucket: Option<u8>,
    #[serde(default)]
    high_memory_pressure_windows: u64,
}

impl JointRuntimeDeviceMeasurements {
    fn observe(&mut self, window: &JointRuntimeDeviceWindowProfile) {
        let Some(observed_tokens_per_second) = window.tokens_per_second() else {
            return;
        };
        self.adaptive_tokens_per_second = Some(match self.adaptive_tokens_per_second {
            Some(previous) => previous * 0.75 + observed_tokens_per_second * 0.25,
            None => observed_tokens_per_second,
        });
        let next_confidence_samples = self.confidence_throughput_samples.saturating_add(1);
        match self.confidence_throughput_mean {
            Some(previous_mean) => {
                let delta = observed_tokens_per_second - previous_mean;
                let next_mean = previous_mean + delta / next_confidence_samples as f64;
                let delta_after = observed_tokens_per_second - next_mean;
                self.confidence_throughput_m2 += delta * delta_after;
                self.confidence_throughput_mean = Some(next_mean);
            }
            None => {
                self.confidence_throughput_mean = Some(observed_tokens_per_second);
                self.confidence_throughput_m2 = 0.0;
            }
        }
        self.confidence_throughput_samples = next_confidence_samples;
        self.windows = self.windows.saturating_add(1);
        self.tokens = self.tokens.saturating_add(window.tokens);
        self.elapsed_seconds += window.elapsed_seconds;
        self.queue_submissions = self
            .queue_submissions
            .saturating_add(window.queue_submissions);
        self.timeline_latency_samples = self
            .timeline_latency_samples
            .saturating_add(window.latency.samples);
        self.timeline_latency_ns_total = self
            .timeline_latency_ns_total
            .saturating_add(window.latency.total_ns);
        self.kernel_profile_samples = self
            .kernel_profile_samples
            .saturating_add(window.latency.kernel_profile_samples);
        self.kernel_dispatches = self
            .kernel_dispatches
            .saturating_add(window.latency.kernel_dispatches);
        self.kernel_gpu_ns_total = self
            .kernel_gpu_ns_total
            .saturating_add(window.latency.kernel_gpu_ns_total);
        if window.shared_submission_arena {
            self.shared_submission_arena_windows =
                self.shared_submission_arena_windows.saturating_add(1);
        }
        if let Some(usage_ratio) = window
            .device_local_usage_ratio
            .filter(|ratio| ratio.is_finite() && *ratio >= 0.0)
        {
            self.adaptive_device_local_usage_ratio = Some(
                self.adaptive_device_local_usage_ratio
                    .map(|previous| previous * 0.75 + usage_ratio * 0.25)
                    .unwrap_or(usage_ratio),
            );
            self.peak_device_local_usage_ratio = Some(
                self.peak_device_local_usage_ratio
                    .map(|previous| previous.max(usage_ratio))
                    .unwrap_or(usage_ratio),
            );
        }
        if let Some(pressure_bucket) = window.device_local_pressure_bucket {
            self.max_device_local_pressure_bucket = Some(
                self.max_device_local_pressure_bucket
                    .map(|previous| previous.max(pressure_bucket))
                    .unwrap_or(pressure_bucket),
            );
            if pressure_bucket >= JOINT_RUNTIME_HIGH_MEMORY_PRESSURE_BUCKET {
                self.high_memory_pressure_windows =
                    self.high_memory_pressure_windows.saturating_add(1);
            }
        }
    }

    fn tokens_per_second(&self) -> Option<f64> {
        (self.tokens != 0 && self.elapsed_seconds.is_finite() && self.elapsed_seconds > 0.0)
            .then(|| self.tokens as f64 / self.elapsed_seconds)
    }

    fn throughput_relative_uncertainty(&self) -> Option<f64> {
        let mean = self
            .confidence_throughput_mean
            .or(self.adaptive_tokens_per_second)
            .or_else(|| self.tokens_per_second())?
            .max(f64::EPSILON);
        let samples = self.confidence_throughput_samples.max(1) as f64;
        let dispersion_relative_standard_error = if self.confidence_throughput_samples > 1 {
            let sample_variance = (self.confidence_throughput_m2
                / (self.confidence_throughput_samples - 1) as f64)
                .max(0.0);
            sample_variance.sqrt() / samples.sqrt() / mean
        } else {
            JOINT_RUNTIME_RELATIVE_NOISE_FLOOR
        };
        let floor = JOINT_RUNTIME_RELATIVE_NOISE_FLOOR / samples.sqrt();
        Some(dispersion_relative_standard_error.max(floor))
    }

    fn confidence_adjusted_tokens_per_second(&self) -> Option<f64> {
        let estimate = self
            .adaptive_tokens_per_second
            .or(self.confidence_throughput_mean)
            .or_else(|| self.tokens_per_second())?
            .max(f64::EPSILON);
        let uncertainty = self.throughput_relative_uncertainty()?;
        Some(
            estimate
                * (1.0 - JOINT_RUNTIME_CONFIDENCE_Z * uncertainty)
                    .clamp(JOINT_RUNTIME_DEVICE_SHARE_MIN_RELATIVE_CAPACITY, 1.0),
        )
    }

    fn average_timeline_latency_ns(&self) -> Option<f64> {
        (self.timeline_latency_samples != 0)
            .then(|| self.timeline_latency_ns_total as f64 / self.timeline_latency_samples as f64)
    }

    fn kernel_gpu_ns_per_token(&self) -> Option<f64> {
        (self.tokens != 0 && self.kernel_gpu_ns_total != 0)
            .then(|| self.kernel_gpu_ns_total as f64 / self.tokens as f64)
    }

    fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "lane_index": self.lane_index,
            "windows": self.windows,
            "tokens": self.tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_per_second": self.tokens_per_second(),
            "adaptive_tokens_per_second": self.adaptive_tokens_per_second,
            "confidence_throughput_samples": self.confidence_throughput_samples,
            "confidence_throughput_mean_tokens_per_second": self.confidence_throughput_mean,
            "throughput_relative_uncertainty": self.throughput_relative_uncertainty(),
            "confidence_adjusted_tokens_per_second": self.confidence_adjusted_tokens_per_second(),
            "queue_submissions": self.queue_submissions,
            "timeline_retirement_latency_samples": self.timeline_latency_samples,
            "timeline_retirement_latency_ns_average": self.average_timeline_latency_ns(),
            "kernel_profile_samples": self.kernel_profile_samples,
            "kernel_dispatches": self.kernel_dispatches,
            "kernel_gpu_ns_total": self.kernel_gpu_ns_total,
            "kernel_gpu_ns_per_token": self.kernel_gpu_ns_per_token(),
            "shared_submission_arena_windows": self.shared_submission_arena_windows,
            "adaptive_device_local_usage_ratio": self.adaptive_device_local_usage_ratio,
            "peak_device_local_usage_ratio": self.peak_device_local_usage_ratio,
            "max_device_local_pressure_bucket": self.max_device_local_pressure_bucket,
            "high_memory_pressure_windows": self.high_memory_pressure_windows,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct JointRuntimeArmMeasurements {
    windows: u64,
    tokens: u64,
    elapsed_seconds: f64,
    queue_submissions: u64,
    timeline_latency_samples: u64,
    timeline_latency_ns_total: u64,
    kernel_profile_samples: u64,
    kernel_dispatches: u64,
    kernel_gpu_ns_total: u64,
    adaptive_tokens_per_second: Option<f64>,
    adaptive_effective_tokens_per_second: Option<f64>,
    adaptive_timeline_latency_ns: Option<f64>,
    adaptive_kernel_gpu_ns_per_token: Option<f64>,
    #[serde(default)]
    adaptive_gradient_reduction_service_seconds: Option<f64>,
    #[serde(default)]
    adaptive_optimizer_boundary_service_seconds: Option<f64>,
    #[serde(default)]
    confidence_effective_samples: u64,
    #[serde(default)]
    confidence_effective_mean: Option<f64>,
    #[serde(default)]
    confidence_effective_m2: f64,
    #[serde(default)]
    throttle_suspect_windows: u64,
    #[serde(default)]
    last_window_throttle_suspect: bool,
    #[serde(default)]
    high_memory_pressure_windows: u64,
    /// Global scored-window ordinal of this arm's most recent clean
    /// observation. This is intentionally persisted so a long-lived training
    /// sidecar can age evidence across restarts without storing every sample.
    #[serde(default)]
    last_observation_ordinal: u64,
    devices: Vec<JointRuntimeDeviceMeasurements>,
}

impl JointRuntimeArmMeasurements {
    #[cfg(test)]
    fn observe(
        &mut self,
        tokens: u64,
        elapsed_seconds: f64,
        queue_submissions: u64,
        latency: SubmissionLatencyDelta,
        device_windows: &[JointRuntimeDeviceWindowProfile],
    ) {
        self.observe_with_phase(
            tokens,
            elapsed_seconds,
            queue_submissions,
            latency,
            device_windows,
            JointRuntimePhaseWindowProfile::default(),
        );
    }

    fn observe_with_phase(
        &mut self,
        tokens: u64,
        elapsed_seconds: f64,
        queue_submissions: u64,
        latency: SubmissionLatencyDelta,
        device_windows: &[JointRuntimeDeviceWindowProfile],
        phase_window: JointRuntimePhaseWindowProfile,
    ) {
        if tokens == 0 || !elapsed_seconds.is_finite() || elapsed_seconds <= 0.0 {
            return;
        }
        let observed_tokens_per_second = tokens as f64 / elapsed_seconds;
        self.adaptive_tokens_per_second = Some(match self.adaptive_tokens_per_second {
            Some(previous) => previous * 0.75 + observed_tokens_per_second * 0.25,
            None => observed_tokens_per_second,
        });
        let heterogeneity_efficiency = if device_windows.is_empty() {
            1.0
        } else {
            let elapsed = device_windows
                .iter()
                .filter_map(|window| {
                    (window.elapsed_seconds.is_finite() && window.elapsed_seconds > 0.0)
                        .then_some(window.elapsed_seconds)
                })
                .collect::<Vec<_>>();
            if elapsed.is_empty() {
                1.0
            } else {
                let slowest = elapsed.iter().copied().fold(0.0f64, f64::max);
                let mean = elapsed.iter().sum::<f64>() / elapsed.len() as f64;
                if slowest > 0.0 {
                    (mean / slowest).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            }
        };
        let effective_tokens_per_second = observed_tokens_per_second * heterogeneity_efficiency;
        let observed_kernel_gpu_ns_per_token = (tokens != 0 && latency.kernel_gpu_ns_total != 0)
            .then(|| latency.kernel_gpu_ns_total as f64 / tokens as f64);
        let throttle_suspect = self.confidence_effective_samples
            >= JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS
            && self
                .adaptive_effective_tokens_per_second
                .zip(self.adaptive_kernel_gpu_ns_per_token)
                .zip(observed_kernel_gpu_ns_per_token)
                .is_some_and(
                    |((baseline_throughput, baseline_gpu_cost), observed_gpu_cost)| {
                        baseline_throughput.is_finite()
                            && baseline_throughput > 0.0
                            && baseline_gpu_cost.is_finite()
                            && baseline_gpu_cost > 0.0
                            && effective_tokens_per_second
                                < baseline_throughput * JOINT_RUNTIME_THROTTLE_THROUGHPUT_RATIO
                            && observed_gpu_cost
                                > baseline_gpu_cost * JOINT_RUNTIME_THROTTLE_GPU_COST_RATIO
                    },
                );
        self.last_window_throttle_suspect = throttle_suspect;
        if throttle_suspect {
            self.throttle_suspect_windows = self.throttle_suspect_windows.saturating_add(1);
        }
        if device_windows.iter().any(|window| {
            window
                .device_local_pressure_bucket
                .is_some_and(|bucket| bucket >= JOINT_RUNTIME_HIGH_MEMORY_PRESSURE_BUCKET)
        }) {
            self.high_memory_pressure_windows = self.high_memory_pressure_windows.saturating_add(1);
        }
        let next_confidence_samples = self.confidence_effective_samples.saturating_add(1);
        match self.confidence_effective_mean {
            Some(previous_mean) => {
                let delta = effective_tokens_per_second - previous_mean;
                let next_mean = previous_mean + delta / next_confidence_samples as f64;
                let delta_after = effective_tokens_per_second - next_mean;
                self.confidence_effective_m2 += delta * delta_after;
                self.confidence_effective_mean = Some(next_mean);
            }
            None => {
                self.confidence_effective_mean = Some(effective_tokens_per_second);
                self.confidence_effective_m2 = 0.0;
            }
        }
        self.confidence_effective_samples = next_confidence_samples;
        self.adaptive_effective_tokens_per_second =
            Some(match self.adaptive_effective_tokens_per_second {
                Some(previous) => previous * 0.75 + effective_tokens_per_second * 0.25,
                None => effective_tokens_per_second,
            });
        if let Some(observed_latency_ns) = latency.average_ns() {
            self.adaptive_timeline_latency_ns = Some(match self.adaptive_timeline_latency_ns {
                Some(previous) => previous * 0.75 + observed_latency_ns * 0.25,
                None => observed_latency_ns,
            });
        }
        if let Some(observed_kernel_gpu_ns_per_token) = observed_kernel_gpu_ns_per_token {
            self.adaptive_kernel_gpu_ns_per_token =
                Some(match self.adaptive_kernel_gpu_ns_per_token {
                    Some(previous) => previous * 0.75 + observed_kernel_gpu_ns_per_token * 0.25,
                    None => observed_kernel_gpu_ns_per_token,
                });
        }
        if phase_window.gradient_reduction_service_seconds.is_finite()
            && phase_window.gradient_reduction_service_seconds >= 0.0
        {
            let observed = phase_window.gradient_reduction_service_seconds;
            self.adaptive_gradient_reduction_service_seconds =
                Some(match self.adaptive_gradient_reduction_service_seconds {
                    Some(previous) => previous * 0.75 + observed * 0.25,
                    None => observed,
                });
        }
        if phase_window.optimizer_boundary_service_seconds.is_finite()
            && phase_window.optimizer_boundary_service_seconds >= 0.0
        {
            let observed = phase_window.optimizer_boundary_service_seconds;
            self.adaptive_optimizer_boundary_service_seconds =
                Some(match self.adaptive_optimizer_boundary_service_seconds {
                    Some(previous) => previous * 0.75 + observed * 0.25,
                    None => observed,
                });
        }
        self.windows = self.windows.saturating_add(1);
        // Direct unit users of this measurement type still get a sensible
        // local ordinal. The joint autotuner overwrites this with its global
        // scored-window ordinal after the observation is committed.
        self.last_observation_ordinal = self.last_observation_ordinal.saturating_add(1);
        self.tokens = self.tokens.saturating_add(tokens);
        self.elapsed_seconds += elapsed_seconds;
        self.queue_submissions = self.queue_submissions.saturating_add(queue_submissions);
        self.timeline_latency_samples = self
            .timeline_latency_samples
            .saturating_add(latency.samples);
        self.timeline_latency_ns_total = self
            .timeline_latency_ns_total
            .saturating_add(latency.total_ns);
        self.kernel_profile_samples = self
            .kernel_profile_samples
            .saturating_add(latency.kernel_profile_samples);
        self.kernel_dispatches = self
            .kernel_dispatches
            .saturating_add(latency.kernel_dispatches);
        self.kernel_gpu_ns_total = self
            .kernel_gpu_ns_total
            .saturating_add(latency.kernel_gpu_ns_total);
        for window in device_windows {
            if let Some(device) = self
                .devices
                .iter_mut()
                .find(|device| device.lane_index == window.lane_index)
            {
                device.observe(window);
            } else {
                let mut device = JointRuntimeDeviceMeasurements {
                    lane_index: window.lane_index,
                    ..JointRuntimeDeviceMeasurements::default()
                };
                device.observe(window);
                self.devices.push(device);
            }
        }
        self.devices.sort_by_key(|device| device.lane_index);
    }

    fn tokens_per_second(&self) -> Option<f64> {
        (self.tokens != 0 && self.elapsed_seconds.is_finite() && self.elapsed_seconds > 0.0)
            .then(|| self.tokens as f64 / self.elapsed_seconds)
    }

    fn average_timeline_latency_ns(&self) -> Option<f64> {
        (self.timeline_latency_samples != 0)
            .then(|| self.timeline_latency_ns_total as f64 / self.timeline_latency_samples as f64)
    }

    fn queue_submissions_per_million_tokens(&self) -> Option<f64> {
        (self.tokens != 0).then(|| self.queue_submissions as f64 * 1_000_000.0 / self.tokens as f64)
    }

    fn kernel_gpu_ns_per_token(&self) -> Option<f64> {
        (self.tokens != 0 && self.kernel_gpu_ns_total != 0)
            .then(|| self.kernel_gpu_ns_total as f64 / self.tokens as f64)
    }

    fn adaptive_effective_tokens_per_second(&self) -> Option<f64> {
        self.adaptive_effective_tokens_per_second
            .or(self.adaptive_tokens_per_second)
            .or(self.confidence_effective_mean)
            .or_else(|| self.tokens_per_second())
    }

    fn observations_since_last_measurement(&self, total_observations: u64) -> u64 {
        total_observations.saturating_sub(self.last_observation_ordinal)
    }

    fn effective_confidence_samples(&self, total_observations: u64) -> f64 {
        if self.confidence_effective_samples == 0 {
            return 0.0;
        }
        let age = self.observations_since_last_measurement(total_observations);
        let decay = JOINT_RUNTIME_OBSERVATION_DECAY.powi(i32::try_from(age).unwrap_or(i32::MAX));
        self.confidence_effective_samples as f64 * decay
    }

    fn relative_uncertainty_at(&self, total_observations: u64) -> Option<f64> {
        let mean = self.adaptive_effective_tokens_per_second()?;
        if mean <= f64::EPSILON {
            return Some(0.0);
        }
        let effective_samples = self
            .effective_confidence_samples(total_observations)
            .max(JOINT_RUNTIME_MIN_EFFECTIVE_SAMPLES);
        let dispersion_relative_standard_error = if self.confidence_effective_samples > 1 {
            let sample_variance = (self.confidence_effective_m2
                / (self.confidence_effective_samples - 1) as f64)
                .max(0.0);
            sample_variance.sqrt() / mean / effective_samples.max(1.0).sqrt()
        } else {
            0.0
        };
        let sampling_floor = JOINT_RUNTIME_RELATIVE_NOISE_FLOOR / effective_samples.sqrt();
        Some(dispersion_relative_standard_error.hypot(sampling_floor))
    }

    fn confidence_adjusted_effective_tokens_per_second_at(
        &self,
        total_observations: u64,
    ) -> Option<f64> {
        let mean = self.adaptive_effective_tokens_per_second()?;
        let relative_uncertainty = self.relative_uncertainty_at(total_observations)?;
        // Preserve the old scheduler's deliberate skepticism of a lucky
        // one-window result while letting the decayed confidence term handle
        // long-run staleness after calibration.
        let calibration = if self.confidence_effective_samples == 1 {
            0.90
        } else {
            1.0
        };
        Some(
            mean * (1.0 - JOINT_RUNTIME_CONFIDENCE_Z * relative_uncertainty).max(0.0) * calibration,
        )
    }

    fn exploration_score_tokens_per_second_at(&self, total_observations: u64) -> Option<f64> {
        let mean = self.adaptive_effective_tokens_per_second()?;
        let relative_uncertainty = self.relative_uncertainty_at(total_observations)?;
        let effective_samples = self
            .effective_confidence_samples(total_observations)
            .max(JOINT_RUNTIME_MIN_EFFECTIVE_SAMPLES);
        let ucb_sampling_bonus = mean
            * JOINT_RUNTIME_UCB_EXPLORATION_SCALE
            * (((total_observations.saturating_add(1)) as f64).ln() / effective_samples).sqrt();
        Some(mean * (1.0 + JOINT_RUNTIME_CONFIDENCE_Z * relative_uncertainty) + ucb_sampling_bonus)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct JointRuntimeProfileKey {
    architecture_revision: String,
    architecture_contract_sha256: Option<String>,
    batch_size: usize,
    gradient_accumulation_steps: usize,
    tokens_per_sequence: usize,
    device_uuids: Vec<String>,
    driver_uuids: Vec<String>,
    transport_backends: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistedJointRuntimeProfile {
    schema_version: u32,
    profile_key: JointRuntimeProfileKey,
    winning_arm: JointRuntimeScheduleArm,
    arms: Vec<JointRuntimeScheduleArm>,
    measurements: Vec<JointRuntimeArmMeasurements>,
}

fn joint_runtime_profile_key(
    config: &ModelConfig,
    args: &Args,
    tokens_per_sequence: usize,
    device_catalog: &[VulkanPhysicalDeviceInfo],
    selected_indices: &[usize],
    transport_backends: &[VulkanGradientTransportBackend],
) -> Result<JointRuntimeProfileKey> {
    let selected_devices = selected_indices
        .iter()
        .map(|index| {
            device_catalog
                .iter()
                .find(|device| device.index == *index)
                .with_context(|| {
                    format!(
                        "Vulkan device {index} disappeared while building persistent joint-runtime profile key"
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(JointRuntimeProfileKey {
        architecture_revision: config.architecture_revision.clone(),
        architecture_contract_sha256: config.architecture_contract_sha256.clone(),
        batch_size: args.batch_size,
        gradient_accumulation_steps: args.gradient_accumulation_steps,
        tokens_per_sequence,
        device_uuids: selected_devices
            .iter()
            .map(|device| device.device_uuid.clone())
            .collect(),
        driver_uuids: selected_devices
            .iter()
            .map(|device| device.driver_uuid.clone())
            .collect(),
        transport_backends: transport_backends
            .iter()
            .map(|backend| backend.label().to_owned())
            .collect(),
    })
}

fn load_joint_runtime_profile(
    model_dir: &Path,
    profile_key: &JointRuntimeProfileKey,
) -> Result<Option<PersistedJointRuntimeProfile>> {
    let path = model_dir.join(JOINT_RUNTIME_PROFILE_FILENAME);
    load_joint_runtime_profile_path(&path, profile_key)
}

fn load_joint_runtime_profile_path(
    path: &Path,
    profile_key: &JointRuntimeProfileKey,
) -> Result<Option<PersistedJointRuntimeProfile>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| {
        format!(
            "reading persistent Vulkan joint-runtime profile {}",
            path.display()
        )
    })?;
    let profile: PersistedJointRuntimeProfile =
        serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "parsing persistent Vulkan joint-runtime profile {}",
                path.display()
            )
        })?;
    if profile.schema_version != JOINT_RUNTIME_PROFILE_SCHEMA_VERSION
        || profile.profile_key != *profile_key
        || profile.arms.len() != profile.measurements.len()
    {
        return Ok(None);
    }
    Ok(Some(profile))
}

fn write_joint_runtime_profile(
    output_dir: &Path,
    profile: &PersistedJointRuntimeProfile,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "creating Vulkan output directory for persistent joint-runtime profile {}",
            output_dir.display()
        )
    })?;
    let path = output_dir.join(JOINT_RUNTIME_PROFILE_FILENAME);
    let encoded = serde_json::to_vec_pretty(profile)
        .context("serializing persistent Vulkan joint-runtime profile")?;
    let temporary_path = output_dir.join(format!(
        ".{JOINT_RUNTIME_PROFILE_FILENAME}.tmp-{}",
        std::process::id()
    ));
    let write_result = (|| -> Result<()> {
        let mut file = File::create(&temporary_path).with_context(|| {
            format!(
                "creating staged persistent Vulkan joint-runtime profile {}",
                temporary_path.display()
            )
        })?;
        file.write_all(&encoded).with_context(|| {
            format!(
                "writing staged persistent Vulkan joint-runtime profile {}",
                temporary_path.display()
            )
        })?;
        file.sync_all().with_context(|| {
            format!(
                "syncing staged persistent Vulkan joint-runtime profile {}",
                temporary_path.display()
            )
        })?;
        fs::rename(&temporary_path, &path).with_context(|| {
            format!(
                "atomically replacing persistent Vulkan joint-runtime profile {}",
                path.display()
            )
        })?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    write_result.with_context(|| {
        format!(
            "writing persistent Vulkan joint-runtime profile {}",
            path.display()
        )
    })?;
    Ok(path)
}

#[derive(Clone, Debug)]
struct JointRuntimeAutotuner {
    arms: Vec<JointRuntimeScheduleArm>,
    measurements: Vec<JointRuntimeArmMeasurements>,
    /// Execution evidence owned by this process only. Persisted measurement
    /// history is intentionally excluded so hardware qualification can prove
    /// which coordinates the current run actually selected and scored.
    current_run_selected_windows: Vec<u64>,
    current_run_scored_windows: Vec<u64>,
    explore_every: u64,
    selection_step: u64,
    last_selected_index: Option<usize>,
    forced_followup_index: Option<usize>,
    warmup_windows_remaining: u8,
    initial_preferred_index: Option<usize>,
    locked_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct JointRuntimeFactorizedScore<T> {
    value: T,
    observed_windows: u64,
    adaptive_score: Option<f64>,
    confidence_adjusted_score: Option<f64>,
    exploration_score: Option<f64>,
    relative_uncertainty: Option<f64>,
    effective_samples: f64,
    observations_since_last_measurement: u64,
}

impl JointRuntimeAutotuner {
    fn candidate_arms(
        baseline_chunk_values: usize,
        baseline_tape_geometry: PhaseAwareTapeGeometry,
        requested_tokens_per_sequence: usize,
    ) -> Vec<JointRuntimeScheduleArm> {
        let mut arms = Vec::with_capacity(10);
        let mut push = |arm: JointRuntimeScheduleArm| {
            if !arms.contains(&arm) {
                arms.push(arm);
            }
        };
        let mut push_overlap_pair = |chunk_values: usize, tape_geometry: PhaseAwareTapeGeometry| {
            push(JointRuntimeScheduleArm {
                gradient_stream_chunk_values: chunk_values.max(1),
                tape_geometry,
                optimizer_broadcast_overlap: true,
            });
            push(JointRuntimeScheduleArm {
                gradient_stream_chunk_values: chunk_values.max(1),
                tape_geometry,
                optimizer_broadcast_overlap: false,
            });
        };

        push_overlap_pair(baseline_chunk_values, baseline_tape_geometry);
        if baseline_chunk_values > 1 {
            push_overlap_pair((baseline_chunk_values / 2).max(1), baseline_tape_geometry);
        }
        if baseline_tape_geometry.sequence_microbatch_size > 1 {
            push_overlap_pair(
                baseline_chunk_values,
                PhaseAwareTapeGeometry {
                    sequence_microbatch_size: (baseline_tape_geometry.sequence_microbatch_size / 2)
                        .max(1),
                    state_checkpoint_stride: baseline_tape_geometry.state_checkpoint_stride,
                },
            );
        }
        if baseline_tape_geometry.state_checkpoint_stride < requested_tokens_per_sequence {
            push_overlap_pair(
                baseline_chunk_values,
                PhaseAwareTapeGeometry {
                    sequence_microbatch_size: baseline_tape_geometry.sequence_microbatch_size,
                    state_checkpoint_stride: baseline_tape_geometry
                        .state_checkpoint_stride
                        .saturating_mul(2)
                        .min(requested_tokens_per_sequence)
                        .max(1),
                },
            );
        }
        if baseline_chunk_values > 1 && baseline_tape_geometry.sequence_microbatch_size > 1 {
            push_overlap_pair(
                (baseline_chunk_values / 2).max(1),
                PhaseAwareTapeGeometry {
                    sequence_microbatch_size: (baseline_tape_geometry.sequence_microbatch_size / 2)
                        .max(1),
                    state_checkpoint_stride: baseline_tape_geometry
                        .state_checkpoint_stride
                        .saturating_mul(2)
                        .min(requested_tokens_per_sequence)
                        .max(1),
                },
            );
        }
        arms
    }

    fn new(
        baseline_chunk_values: usize,
        baseline_tape_geometry: PhaseAwareTapeGeometry,
        requested_tokens_per_sequence: usize,
        persisted: Option<&PersistedJointRuntimeProfile>,
        lock_persisted_winner: bool,
    ) -> Option<Self> {
        if std::env::var_os(HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_DISABLE_ENV).is_some() {
            return None;
        }
        let explore_every = match std::env::var(HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_EXPLORE_EVERY_ENV)
        {
            Ok(value) => value
                .parse::<u64>()
                .ok()
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_JOINT_RUNTIME_AUTOTUNE_EXPLORE_EVERY),
            Err(_) => DEFAULT_JOINT_RUNTIME_AUTOTUNE_EXPLORE_EVERY,
        };
        let mut arms = Self::candidate_arms(
            baseline_chunk_values,
            baseline_tape_geometry,
            requested_tokens_per_sequence,
        );
        if let Some(profile) = persisted {
            let winner = profile.winning_arm;
            let safe_winner = winner.gradient_stream_chunk_values <= baseline_chunk_values
                && winner.tape_geometry.sequence_microbatch_size > 0
                && winner.tape_geometry.sequence_microbatch_size
                    <= baseline_tape_geometry.sequence_microbatch_size
                && winner.tape_geometry.state_checkpoint_stride
                    >= baseline_tape_geometry.state_checkpoint_stride
                && winner.tape_geometry.state_checkpoint_stride <= requested_tokens_per_sequence;
            if safe_winner && !arms.contains(&winner) {
                arms.push(winner);
            }
        }

        let mut measurements = vec![JointRuntimeArmMeasurements::default(); arms.len()];
        let mut initial_preferred_index = None;
        let mut locked_index = None;
        if let Some(profile) = persisted {
            for (stored_arm, stored_measurements) in
                profile.arms.iter().zip(profile.measurements.iter())
            {
                if let Some(index) = arms.iter().position(|arm| arm == stored_arm) {
                    measurements[index] = stored_measurements.clone();
                }
            }
            if let Some(index) = arms.iter().position(|arm| *arm == profile.winning_arm) {
                if index != 0 {
                    arms.swap(0, index);
                    measurements.swap(0, index);
                }
                initial_preferred_index = Some(0);
                if lock_persisted_winner {
                    locked_index = Some(0);
                }
            }
        }
        let total_stored_observations = measurements
            .iter()
            .map(|measurements| measurements.windows)
            .sum::<u64>();
        // Profiles written before stale-arm ordinals existed deserialize with
        // zero here. Treat that imported evidence as fresh at load time, then
        // age it normally as this run accumulates new scored windows.
        for measurement in &mut measurements {
            if measurement.windows != 0 && measurement.last_observation_ordinal == 0 {
                measurement.last_observation_ordinal = total_stored_observations;
            }
        }
        Some(Self {
            current_run_selected_windows: vec![0; arms.len()],
            current_run_scored_windows: vec![0; arms.len()],
            arms,
            measurements,
            explore_every,
            selection_step: 0,
            last_selected_index: None,
            forced_followup_index: None,
            warmup_windows_remaining: 0,
            initial_preferred_index,
            locked_index,
        })
    }

    fn locked_arm(&self) -> Option<JointRuntimeScheduleArm> {
        self.locked_index
            .and_then(|index| self.arms.get(index).copied())
    }

    fn total_scored_observations(&self) -> u64 {
        self.measurements
            .iter()
            .map(|measurements| measurements.windows)
            .sum()
    }

    fn unique_factor_values<T: Copy + Eq>(
        &self,
        value_for_arm: impl Fn(JointRuntimeScheduleArm) -> T,
    ) -> Vec<T> {
        let mut values = Vec::new();
        for arm in self.arms.iter().copied() {
            let value = value_for_arm(arm);
            if !values.contains(&value) {
                values.push(value);
            }
        }
        values
    }

    /// Collapse the persisted composite-arm history onto one independently
    /// controlled scheduler dimension. Old observations remain useful even
    /// when the other dimensions differ, which is the key property that avoids
    /// relearning the full transport x overlap x tape Cartesian product.
    fn factorized_scores<T: Copy + Eq>(
        &self,
        candidates: &[T],
        value_for_arm: impl Fn(JointRuntimeScheduleArm) -> T,
        score_for_measurement: impl Fn(&JointRuntimeArmMeasurements) -> Option<f64>,
    ) -> Vec<JointRuntimeFactorizedScore<T>> {
        let total_observations = self.total_scored_observations();
        candidates
            .iter()
            .copied()
            .map(|candidate| {
                let mut observed_windows = 0u64;
                let mut effective_samples = 0.0f64;
                let mut weighted_score = 0.0f64;
                let mut weighted_score_squared = 0.0f64;
                let mut last_observation_ordinal = 0u64;

                for (arm, measurements) in self.arms.iter().copied().zip(&self.measurements) {
                    if value_for_arm(arm) != candidate || measurements.windows == 0 {
                        continue;
                    }
                    let Some(score) = score_for_measurement(measurements)
                        .filter(|score| score.is_finite() && *score > 0.0)
                    else {
                        continue;
                    };
                    let age = measurements.observations_since_last_measurement(total_observations);
                    let decay = JOINT_RUNTIME_OBSERVATION_DECAY
                        .powi(i32::try_from(age).unwrap_or(i32::MAX));
                    let weight = measurements.windows as f64 * decay;
                    if !weight.is_finite() || weight <= 0.0 {
                        continue;
                    }
                    observed_windows = observed_windows.saturating_add(measurements.windows);
                    effective_samples += weight;
                    weighted_score += score * weight;
                    weighted_score_squared += score * score * weight;
                    last_observation_ordinal =
                        last_observation_ordinal.max(measurements.last_observation_ordinal);
                }

                let observations_since_last_measurement = if observed_windows == 0 {
                    total_observations
                } else {
                    total_observations.saturating_sub(last_observation_ordinal)
                };
                let adaptive_score = (effective_samples > 0.0)
                    .then(|| weighted_score / effective_samples)
                    .filter(|score| score.is_finite() && *score > 0.0);
                let (confidence_adjusted_score, exploration_score, relative_uncertainty) =
                    if let Some(mean) = adaptive_score {
                        let effective = effective_samples.max(JOINT_RUNTIME_MIN_EFFECTIVE_SAMPLES);
                        let variance =
                            (weighted_score_squared / effective_samples - mean * mean).max(0.0);
                        let dispersion_relative_standard_error =
                            variance.sqrt() / mean / effective.max(1.0).sqrt();
                        let sampling_floor = JOINT_RUNTIME_RELATIVE_NOISE_FLOOR / effective.sqrt();
                        let relative_uncertainty =
                            dispersion_relative_standard_error.hypot(sampling_floor);
                        let confidence = mean
                            * (1.0 - JOINT_RUNTIME_CONFIDENCE_Z * relative_uncertainty).max(0.0);
                        let ucb_sampling_bonus = mean
                            * JOINT_RUNTIME_UCB_EXPLORATION_SCALE
                            * (((total_observations.saturating_add(1)) as f64).ln() / effective)
                                .sqrt();
                        let exploration = mean
                            * (1.0 + JOINT_RUNTIME_CONFIDENCE_Z * relative_uncertainty)
                            + ucb_sampling_bonus;
                        (
                            Some(confidence),
                            Some(exploration),
                            Some(relative_uncertainty),
                        )
                    } else {
                        (None, None, None)
                    };

                JointRuntimeFactorizedScore {
                    value: candidate,
                    observed_windows,
                    adaptive_score,
                    confidence_adjusted_score,
                    exploration_score,
                    relative_uncertainty,
                    effective_samples,
                    observations_since_last_measurement,
                }
            })
            .collect()
    }

    fn tape_sequence_microbatch_scores(&self) -> Vec<JointRuntimeFactorizedScore<usize>> {
        let candidates =
            self.unique_factor_values(|arm| arm.tape_geometry.sequence_microbatch_size);
        self.factorized_scores(
            &candidates,
            |arm| arm.tape_geometry.sequence_microbatch_size,
            JointRuntimeArmMeasurements::adaptive_effective_tokens_per_second,
        )
    }

    fn tape_checkpoint_stride_scores(&self) -> Vec<JointRuntimeFactorizedScore<usize>> {
        let candidates = self.unique_factor_values(|arm| arm.tape_geometry.state_checkpoint_stride);
        self.factorized_scores(
            &candidates,
            |arm| arm.tape_geometry.state_checkpoint_stride,
            JointRuntimeArmMeasurements::adaptive_effective_tokens_per_second,
        )
    }

    fn gradient_transport_scores(&self) -> Vec<JointRuntimeFactorizedScore<usize>> {
        let candidates = self.unique_factor_values(|arm| arm.gradient_stream_chunk_values);
        self.factorized_scores(
            &candidates,
            |arm| arm.gradient_stream_chunk_values,
            |measurements| {
                measurements
                    .adaptive_gradient_reduction_service_seconds
                    .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
                    .map(|seconds| 1.0 / seconds)
            },
        )
    }

    fn optimizer_overlap_scores(&self) -> Vec<JointRuntimeFactorizedScore<bool>> {
        let candidates = self.unique_factor_values(|arm| arm.optimizer_broadcast_overlap);
        self.factorized_scores(
            &candidates,
            |arm| arm.optimizer_broadcast_overlap,
            |measurements| {
                measurements
                    .adaptive_optimizer_boundary_service_seconds
                    .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
                    .map(|seconds| 1.0 / seconds)
            },
        )
    }

    fn choose_factorized_value<T: Copy + Eq>(
        scores: &[JointRuntimeFactorizedScore<T>],
        bootstrap: bool,
        explore: bool,
    ) -> Option<T> {
        if scores.iter().all(|score| score.adaptive_score.is_none()) {
            return scores.first().map(|score| score.value);
        }
        if bootstrap {
            if let Some(score) = scores
                .iter()
                .find(|score| score.observed_windows < JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS)
            {
                return Some(score.value);
            }
        }
        let ranked = scores.iter().max_by(|left, right| {
            let left_primary = if explore {
                left.exploration_score
            } else {
                left.confidence_adjusted_score
            }
            .unwrap_or(0.0);
            let right_primary = if explore {
                right.exploration_score
            } else {
                right.confidence_adjusted_score
            }
            .unwrap_or(0.0);
            left_primary
                .total_cmp(&right_primary)
                .then_with(|| {
                    left.adaptive_score
                        .unwrap_or(0.0)
                        .total_cmp(&right.adaptive_score.unwrap_or(0.0))
                })
                .then_with(|| {
                    left.observations_since_last_measurement
                        .cmp(&right.observations_since_last_measurement)
                })
        });
        ranked
            .map(|score| score.value)
            .or_else(|| scores.first().map(|score| score.value))
    }

    /// Return a dimensionless urgency signal for an independently controlled
    /// phase selector. Bootstrap debt is kept separate from UCB pressure so
    /// factors with different score units (service rate vs. tokens/sec) can
    /// still compete for the next probe without comparing raw magnitudes.
    fn factorized_selector_priority<T: Copy + Eq>(
        scores: &[JointRuntimeFactorizedScore<T>],
    ) -> (u64, f64, u64) {
        if scores.is_empty() {
            return (0, 0.0, 0);
        }
        let minimum_observed_windows = scores
            .iter()
            .map(|score| score.observed_windows)
            .min()
            .unwrap_or(0);
        let bootstrap_debt =
            JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS.saturating_sub(minimum_observed_windows);
        let best_exploration = scores
            .iter()
            .filter_map(|score| score.exploration_score)
            .filter(|score| score.is_finite() && *score > 0.0)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        let best_confidence = scores
            .iter()
            .filter_map(|score| score.confidence_adjusted_score)
            .filter(|score| score.is_finite() && *score > 0.0)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        let exploration_pressure = if best_confidence > 0.0 {
            (best_exploration / best_confidence).max(1.0)
        } else if best_exploration > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let staleness = scores
            .iter()
            .map(|score| score.observations_since_last_measurement)
            .max()
            .unwrap_or(0);
        (bootstrap_debt, exploration_pressure, staleness)
    }

    fn active_factorized_selector(
        step: u64,
        explore: bool,
        gradient_scores: &[JointRuntimeFactorizedScore<usize>],
        optimizer_scores: &[JointRuntimeFactorizedScore<bool>],
        tape_microbatch_scores: &[JointRuntimeFactorizedScore<usize>],
        tape_stride_scores: &[JointRuntimeFactorizedScore<usize>],
    ) -> Option<usize> {
        let priorities = [
            Self::factorized_selector_priority(gradient_scores),
            Self::factorized_selector_priority(optimizer_scores),
            Self::factorized_selector_priority(tape_microbatch_scores),
            Self::factorized_selector_priority(tape_stride_scores),
        ];
        let max_bootstrap_debt = priorities
            .iter()
            .map(|priority| priority.0)
            .max()
            .unwrap_or(0);
        if max_bootstrap_debt > 0 {
            let tied = priorities
                .iter()
                .enumerate()
                .filter_map(|(index, priority)| (priority.0 == max_bootstrap_debt).then_some(index))
                .collect::<Vec<_>>();
            return tied.get(step as usize % tied.len()).copied();
        }
        if !explore {
            return None;
        }
        priorities
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.2.cmp(&right.2))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
    }

    fn best_gradient_transport_chunk_values(&self) -> Option<usize> {
        Self::choose_factorized_value(&self.gradient_transport_scores(), false, false)
    }

    fn best_optimizer_broadcast_overlap(&self) -> Option<bool> {
        Self::choose_factorized_value(&self.optimizer_overlap_scores(), false, false)
    }

    fn best_tape_geometry(&self) -> Option<PhaseAwareTapeGeometry> {
        Some(PhaseAwareTapeGeometry {
            sequence_microbatch_size: Self::choose_factorized_value(
                &self.tape_sequence_microbatch_scores(),
                false,
                false,
            )?,
            state_checkpoint_stride: Self::choose_factorized_value(
                &self.tape_checkpoint_stride_scores(),
                false,
                false,
            )?,
        })
    }

    fn ensure_arm(&mut self, arm: JointRuntimeScheduleArm) -> usize {
        if let Some(index) = self.arms.iter().position(|candidate| *candidate == arm) {
            return index;
        }
        self.arms.push(arm);
        self.measurements
            .push(JointRuntimeArmMeasurements::default());
        self.current_run_selected_windows.push(0);
        self.current_run_scored_windows.push(0);
        self.arms.len() - 1
    }

    fn compare_measurements(
        left: &JointRuntimeArmMeasurements,
        right: &JointRuntimeArmMeasurements,
        total_observations: u64,
    ) -> std::cmp::Ordering {
        let left_confidence = left
            .confidence_adjusted_effective_tokens_per_second_at(total_observations)
            .unwrap_or(0.0);
        let right_confidence = right
            .confidence_adjusted_effective_tokens_per_second_at(total_observations)
            .unwrap_or(0.0);
        let left_effective = left
            .adaptive_effective_tokens_per_second
            .or(left.adaptive_tokens_per_second)
            .or_else(|| left.tokens_per_second())
            .unwrap_or(0.0);
        let right_effective = right
            .adaptive_effective_tokens_per_second
            .or(right.adaptive_tokens_per_second)
            .or_else(|| right.tokens_per_second())
            .unwrap_or(0.0);
        let left_tps = left
            .adaptive_tokens_per_second
            .or_else(|| left.tokens_per_second())
            .unwrap_or(0.0);
        let right_tps = right
            .adaptive_tokens_per_second
            .or_else(|| right.tokens_per_second())
            .unwrap_or(0.0);
        let left_kernel = left
            .adaptive_kernel_gpu_ns_per_token
            .or_else(|| left.kernel_gpu_ns_per_token())
            .unwrap_or(f64::INFINITY);
        let right_kernel = right
            .adaptive_kernel_gpu_ns_per_token
            .or_else(|| right.kernel_gpu_ns_per_token())
            .unwrap_or(f64::INFINITY);
        let left_latency = left
            .adaptive_timeline_latency_ns
            .or_else(|| left.average_timeline_latency_ns())
            .unwrap_or(f64::INFINITY);
        let right_latency = right
            .adaptive_timeline_latency_ns
            .or_else(|| right.average_timeline_latency_ns())
            .unwrap_or(f64::INFINITY);
        left_confidence
            .total_cmp(&right_confidence)
            .then_with(|| left_effective.total_cmp(&right_effective))
            .then_with(|| left_tps.total_cmp(&right_tps))
            .then_with(|| right_kernel.total_cmp(&left_kernel))
            .then_with(|| right_latency.total_cmp(&left_latency))
    }

    fn best_measured_index(&self) -> Option<usize> {
        let total_observations = self.total_scored_observations();
        self.measurements
            .iter()
            .enumerate()
            .filter(|(_, measurements)| measurements.windows != 0)
            .max_by(|(_, left), (_, right)| {
                Self::compare_measurements(left, right, total_observations)
            })
            .map(|(index, _)| index)
    }

    #[cfg(test)]
    fn best_exploration_index(&self) -> Option<usize> {
        let total_observations = self.total_scored_observations();
        self.measurements
            .iter()
            .enumerate()
            .filter(|(_, measurements)| measurements.windows != 0)
            .max_by(|(_, left), (_, right)| {
                left.exploration_score_tokens_per_second_at(total_observations)
                    .unwrap_or(0.0)
                    .total_cmp(
                        &right
                            .exploration_score_tokens_per_second_at(total_observations)
                            .unwrap_or(0.0),
                    )
                    .then_with(|| {
                        left.observations_since_last_measurement(total_observations)
                            .cmp(&right.observations_since_last_measurement(total_observations))
                    })
            })
            .map(|(index, _)| index)
    }

    /// Convert persisted per-lane throughput into relative scheduling capacity
    /// for the currently selected joint arm. Current-arm evidence wins; missing
    /// lanes borrow same-device evidence from other arms so schedule switches do
    /// not throw away everything learned about a heterogeneous adapter. Each
    /// lane contributes a one-sided confidence-adjusted throughput estimate, so
    /// high-variance adapters cannot win shard share from a lucky EWMA spike.
    /// A short confidence ramp still keeps the first few samples close to the
    /// old equal-share behavior while allowing stable long runs to converge.
    fn lane_workload_weights(&self, selected_index: usize, lane_count: usize) -> Vec<f64> {
        if lane_count == 0 {
            return Vec::new();
        }
        let mut throughput = vec![None; lane_count];
        let mut sample_windows = vec![0u64; lane_count];

        if let Some(selected) = self.measurements.get(selected_index) {
            for device in &selected.devices {
                if device.lane_index >= lane_count {
                    continue;
                }
                let estimate = device
                    .confidence_adjusted_tokens_per_second()
                    .or(device.adaptive_tokens_per_second)
                    .or_else(|| device.tokens_per_second())
                    .filter(|value| value.is_finite() && *value > 0.0);
                if let Some(estimate) = estimate {
                    throughput[device.lane_index] = Some(estimate);
                    sample_windows[device.lane_index] = device.windows;
                }
            }
        }

        for lane_index in 0..lane_count {
            if throughput[lane_index].is_some() {
                continue;
            }
            let mut weighted_throughput = 0.0f64;
            let mut total_windows = 0u64;
            for measurements in &self.measurements {
                let Some(device) = measurements
                    .devices
                    .iter()
                    .find(|device| device.lane_index == lane_index)
                else {
                    continue;
                };
                let Some(estimate) = device
                    .confidence_adjusted_tokens_per_second()
                    .or(device.adaptive_tokens_per_second)
                    .or_else(|| device.tokens_per_second())
                    .filter(|value| value.is_finite() && *value > 0.0)
                else {
                    continue;
                };
                let windows = device.windows.max(1);
                weighted_throughput += estimate * windows as f64;
                total_windows = total_windows.saturating_add(windows);
            }
            if total_windows != 0 {
                throughput[lane_index] = Some(weighted_throughput / total_windows as f64);
                sample_windows[lane_index] = total_windows;
            }
        }

        let observed = throughput
            .iter()
            .flatten()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .collect::<Vec<_>>();
        if observed.len() < 2 {
            return vec![1.0; lane_count];
        }
        let reference = observed.iter().sum::<f64>() / observed.len() as f64;
        if !reference.is_finite() || reference <= f64::EPSILON {
            return vec![1.0; lane_count];
        }

        throughput
            .into_iter()
            .zip(sample_windows)
            .map(|(estimate, windows)| {
                let Some(estimate) = estimate else {
                    return 1.0;
                };
                let relative = (estimate / reference).clamp(
                    JOINT_RUNTIME_DEVICE_SHARE_MIN_RELATIVE_CAPACITY,
                    JOINT_RUNTIME_DEVICE_SHARE_MAX_RELATIVE_CAPACITY,
                );
                let confidence = (windows as f64 / JOINT_RUNTIME_DEVICE_SHARE_CONFIDENCE_WINDOWS)
                    .clamp(0.0, 1.0);
                1.0 + (relative - 1.0) * confidence
            })
            .collect()
    }

    fn persisted_profile(
        &self,
        profile_key: JointRuntimeProfileKey,
    ) -> Option<PersistedJointRuntimeProfile> {
        let winning_index = self.locked_index.or_else(|| self.best_measured_index())?;
        let mut winning_arm = self.arms[winning_index];
        if self.locked_index.is_none() {
            winning_arm.gradient_stream_chunk_values = self
                .best_gradient_transport_chunk_values()
                .unwrap_or(winning_arm.gradient_stream_chunk_values);
            winning_arm.optimizer_broadcast_overlap = self
                .best_optimizer_broadcast_overlap()
                .unwrap_or(winning_arm.optimizer_broadcast_overlap);
            winning_arm.tape_geometry = self
                .best_tape_geometry()
                .unwrap_or(winning_arm.tape_geometry);
        }
        Some(PersistedJointRuntimeProfile {
            schema_version: JOINT_RUNTIME_PROFILE_SCHEMA_VERSION,
            profile_key,
            winning_arm,
            arms: self.arms.clone(),
            measurements: self.measurements.clone(),
        })
    }

    /// Independently select gradient transport width, optimizer/broadcast
    /// overlap, tape sequence microbatch, and tape checkpoint stride, then
    /// materialize their product only for the window that will actually
    /// execute. This turns the old global arm table into an observation store
    /// rather than a Cartesian control surface, including inside tape geometry.
    ///
    /// Only one factor is allowed to bootstrap/explore on a selection step;
    /// the other two exploit their marginal evidence. Every resulting composite
    /// switch still gets one unscored dwell window so prior broadcasts retire
    /// and transport staging reaches the requested width before measurement.
    fn select(&mut self) -> (usize, JointRuntimeScheduleArm, bool) {
        let step = self.selection_step;
        self.selection_step = self.selection_step.wrapping_add(1);
        let mut warmup_window = false;
        let index = if let Some(index) = self.locked_index {
            self.forced_followup_index = None;
            self.initial_preferred_index = None;
            if self.last_selected_index == Some(index) && self.warmup_windows_remaining != 0 {
                self.warmup_windows_remaining -= 1;
                warmup_window = true;
            }
            index
        } else if let Some(index) = self.forced_followup_index.take() {
            if self.warmup_windows_remaining != 0 {
                self.warmup_windows_remaining -= 1;
                warmup_window = true;
                self.forced_followup_index = Some(index);
            }
            index
        } else if let Some(index) = self.initial_preferred_index.take() {
            index
        } else {
            // Keep the old aggregate exploration cadence, but let bootstrap
            // debt and normalized UCB pressure decide which phase gets the
            // probe instead of assigning fixed transport/optimizer/tape turns.
            let selector_round = step / 4;
            let explore = selector_round != 0 && selector_round % self.explore_every == 0;
            let gradient_scores = self.gradient_transport_scores();
            let optimizer_scores = self.optimizer_overlap_scores();
            let tape_microbatch_scores = self.tape_sequence_microbatch_scores();
            let tape_stride_scores = self.tape_checkpoint_stride_scores();
            let active_factor = Self::active_factorized_selector(
                step,
                explore,
                &gradient_scores,
                &optimizer_scores,
                &tape_microbatch_scores,
                &tape_stride_scores,
            );
            let fallback = self
                .best_measured_index()
                .and_then(|index| self.arms.get(index).copied())
                .unwrap_or(self.arms[0]);
            let arm = JointRuntimeScheduleArm {
                gradient_stream_chunk_values: Self::choose_factorized_value(
                    &gradient_scores,
                    active_factor == Some(0),
                    active_factor == Some(0) && explore,
                )
                .unwrap_or(fallback.gradient_stream_chunk_values),
                optimizer_broadcast_overlap: Self::choose_factorized_value(
                    &optimizer_scores,
                    active_factor == Some(1),
                    active_factor == Some(1) && explore,
                )
                .unwrap_or(fallback.optimizer_broadcast_overlap),
                tape_geometry: PhaseAwareTapeGeometry {
                    sequence_microbatch_size: Self::choose_factorized_value(
                        &tape_microbatch_scores,
                        active_factor == Some(2),
                        active_factor == Some(2) && explore,
                    )
                    .unwrap_or(fallback.tape_geometry.sequence_microbatch_size),
                    state_checkpoint_stride: Self::choose_factorized_value(
                        &tape_stride_scores,
                        active_factor == Some(3),
                        active_factor == Some(3) && explore,
                    )
                    .unwrap_or(fallback.tape_geometry.state_checkpoint_stride),
                },
            };
            self.ensure_arm(arm)
        };
        let switched = self.last_selected_index != Some(index);
        if switched {
            self.forced_followup_index = Some(index);
            self.warmup_windows_remaining =
                JOINT_RUNTIME_WARMUP_WINDOWS_AFTER_SWITCH.saturating_sub(1);
            warmup_window = true;
        }
        self.last_selected_index = Some(index);
        self.current_run_selected_windows[index] =
            self.current_run_selected_windows[index].saturating_add(1);
        (index, self.arms[index], !warmup_window)
    }

    fn observe_with_phase(
        &mut self,
        index: usize,
        tokens: u64,
        elapsed_seconds: f64,
        queue_submissions: u64,
        latency: SubmissionLatencyDelta,
        device_windows: &[JointRuntimeDeviceWindowProfile],
        phase_window: JointRuntimePhaseWindowProfile,
    ) {
        if index < self.measurements.len() {
            let windows_before = self.measurements[index].windows;
            self.measurements[index].observe_with_phase(
                tokens,
                elapsed_seconds,
                queue_submissions,
                latency,
                device_windows,
                phase_window,
            );
            if self.measurements[index].windows > windows_before {
                self.current_run_scored_windows[index] =
                    self.current_run_scored_windows[index].saturating_add(1);
            }
            let observation_ordinal = self.total_scored_observations();
            self.measurements[index].last_observation_ordinal = observation_ordinal;
            if std::env::var_os(HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_LOG_ENV).is_some() {
                let arm = self.arms[index];
                let measurements = &self.measurements[index];
                let lane_count = device_windows
                    .iter()
                    .map(|window| window.lane_index)
                    .max()
                    .map(|index| index + 1)
                    .unwrap_or(0);
                let learned_device_workload_weights = self.lane_workload_weights(index, lane_count);
                let peak_device_local_usage_ratio = device_windows
                    .iter()
                    .filter_map(|window| window.device_local_usage_ratio)
                    .filter(|ratio| ratio.is_finite())
                    .reduce(f64::max);
                let max_device_local_pressure_bucket = device_windows
                    .iter()
                    .filter_map(|window| window.device_local_pressure_bucket)
                    .max();
                eprintln!(
                    "vulkan_joint_runtime_autotune arm={} chunk_values={} tape_microbatch={} checkpoint_stride={} optimizer_broadcast_overlap={} windows={} observed_tokens_per_second={:.3} adaptive_tokens_per_second={:.3} adaptive_effective_tokens_per_second={:.3} confidence_adjusted_effective_tokens_per_second={:.3} exploration_score_tokens_per_second={:.3} effective_confidence_samples={:.3} observations_since_last_measurement={} timeline_retirement_samples={} timeline_retirement_average_ns={:.1} kernel_profile_samples={} kernel_dispatches={} kernel_gpu_ns_per_token={:.3} device_lanes={} learned_device_workload_weights={:?} peak_device_local_usage_ratio={:.3} max_device_local_pressure_bucket={} throttle_suspect={} throttle_suspect_windows={} gradient_reduction_service_ms={:.3} adaptive_gradient_reduction_service_ms={:.3} optimizer_boundary_service_ms={:.3} adaptive_optimizer_boundary_service_ms={:.3} queue_submissions_per_million_tokens={:.3}",
                    index,
                    arm.gradient_stream_chunk_values,
                    arm.tape_geometry.sequence_microbatch_size,
                    arm.tape_geometry.state_checkpoint_stride,
                    arm.optimizer_broadcast_overlap,
                    measurements.windows,
                    tokens as f64 / elapsed_seconds,
                    measurements.adaptive_tokens_per_second.unwrap_or(0.0),
                    measurements
                        .adaptive_effective_tokens_per_second
                        .unwrap_or(0.0),
                    measurements
                        .confidence_adjusted_effective_tokens_per_second_at(observation_ordinal)
                        .unwrap_or(0.0),
                    measurements
                        .exploration_score_tokens_per_second_at(observation_ordinal)
                        .unwrap_or(0.0),
                    measurements.effective_confidence_samples(observation_ordinal),
                    measurements.observations_since_last_measurement(observation_ordinal),
                    latency.samples,
                    latency.average_ns().unwrap_or(0.0),
                    latency.kernel_profile_samples,
                    latency.kernel_dispatches,
                    measurements.kernel_gpu_ns_per_token().unwrap_or(0.0),
                    device_windows.len(),
                    learned_device_workload_weights,
                    peak_device_local_usage_ratio.unwrap_or(0.0),
                    max_device_local_pressure_bucket.unwrap_or(0),
                    measurements.last_window_throttle_suspect,
                    measurements.throttle_suspect_windows,
                    phase_window.gradient_reduction_service_seconds * 1_000.0,
                    measurements
                        .adaptive_gradient_reduction_service_seconds
                        .unwrap_or(0.0)
                        * 1_000.0,
                    phase_window.optimizer_boundary_service_seconds * 1_000.0,
                    measurements
                        .adaptive_optimizer_boundary_service_seconds
                        .unwrap_or(0.0)
                        * 1_000.0,
                    measurements
                        .queue_submissions_per_million_tokens()
                        .unwrap_or(0.0),
                );
            }
        }
    }

    fn report(&self) -> serde_json::Value {
        let total_observations = self.total_scored_observations();
        let gradient_scores = self.gradient_transport_scores();
        let optimizer_scores = self.optimizer_overlap_scores();
        let tape_microbatch_scores = self.tape_sequence_microbatch_scores();
        let tape_stride_scores = self.tape_checkpoint_stride_scores();
        let gradient_priority = Self::factorized_selector_priority(&gradient_scores);
        let optimizer_priority = Self::factorized_selector_priority(&optimizer_scores);
        let tape_microbatch_priority = Self::factorized_selector_priority(&tape_microbatch_scores);
        let tape_stride_priority = Self::factorized_selector_priority(&tape_stride_scores);
        let phase_selectors = serde_json::json!({
            "gradient_transport": gradient_scores.iter().map(|score| serde_json::json!({
                "gradient_stream_chunk_values": score.value,
                "observed_windows": score.observed_windows,
                "adaptive_service_seconds": score.adaptive_score.map(|rate| 1.0 / rate),
                "confidence_adjusted_service_rate": score.confidence_adjusted_score,
                "exploration_service_rate": score.exploration_score,
                "relative_uncertainty": score.relative_uncertainty,
                "effective_samples": score.effective_samples,
                "observations_since_last_measurement": score.observations_since_last_measurement,
            })).collect::<Vec<_>>(),
            "optimizer_broadcast_overlap": optimizer_scores.iter().map(|score| serde_json::json!({
                "enabled": score.value,
                "observed_windows": score.observed_windows,
                "adaptive_service_seconds": score.adaptive_score.map(|rate| 1.0 / rate),
                "confidence_adjusted_service_rate": score.confidence_adjusted_score,
                "exploration_service_rate": score.exploration_score,
                "relative_uncertainty": score.relative_uncertainty,
                "effective_samples": score.effective_samples,
                "observations_since_last_measurement": score.observations_since_last_measurement,
            })).collect::<Vec<_>>(),
            "tape_sequence_microbatch": tape_microbatch_scores.iter().map(|score| serde_json::json!({
                "sequence_microbatch_size": score.value,
                "observed_windows": score.observed_windows,
                "adaptive_effective_tokens_per_second": score.adaptive_score,
                "confidence_adjusted_effective_tokens_per_second": score.confidence_adjusted_score,
                "exploration_score_tokens_per_second": score.exploration_score,
                "relative_uncertainty": score.relative_uncertainty,
                "effective_samples": score.effective_samples,
                "observations_since_last_measurement": score.observations_since_last_measurement,
            })).collect::<Vec<_>>(),
            "tape_checkpoint_stride": tape_stride_scores.iter().map(|score| serde_json::json!({
                "state_checkpoint_stride": score.value,
                "observed_windows": score.observed_windows,
                "adaptive_effective_tokens_per_second": score.adaptive_score,
                "confidence_adjusted_effective_tokens_per_second": score.confidence_adjusted_score,
                "exploration_score_tokens_per_second": score.exploration_score,
                "relative_uncertainty": score.relative_uncertainty,
                "effective_samples": score.effective_samples,
                "observations_since_last_measurement": score.observations_since_last_measurement,
            })).collect::<Vec<_>>(),
            "selected_gradient_stream_chunk_values": self.best_gradient_transport_chunk_values(),
            "selected_optimizer_broadcast_overlap": self.best_optimizer_broadcast_overlap(),
            "selected_tape_geometry": self.best_tape_geometry().map(|geometry| serde_json::json!({
                "sequence_microbatch_size": geometry.sequence_microbatch_size,
                "state_checkpoint_stride": geometry.state_checkpoint_stride,
            })),
            "arbiter": {
                "policy": "bootstrap-debt-then-normalized-decayed-ucb-v1",
                "gradient_transport": {
                    "bootstrap_debt": gradient_priority.0,
                    "normalized_ucb_pressure": gradient_priority.1,
                    "max_staleness": gradient_priority.2,
                },
                "optimizer_broadcast_overlap": {
                    "bootstrap_debt": optimizer_priority.0,
                    "normalized_ucb_pressure": optimizer_priority.1,
                    "max_staleness": optimizer_priority.2,
                },
                "tape_sequence_microbatch": {
                    "bootstrap_debt": tape_microbatch_priority.0,
                    "normalized_ucb_pressure": tape_microbatch_priority.1,
                    "max_staleness": tape_microbatch_priority.2,
                },
                "tape_checkpoint_stride": {
                    "bootstrap_debt": tape_stride_priority.0,
                    "normalized_ucb_pressure": tape_stride_priority.1,
                    "max_staleness": tape_stride_priority.2,
                },
            },
            "control_mode": "factorized-decayed-ucb-authoritative-v4-steady-window-quality",
        });
        let mut report = serde_json::json!({
            "scheduler": "joint-runtime-throughput-v9-steady-window-quality-factorized-decayed-ucb-persistent-heterogeneous-device-shares",
            "explore_every": self.explore_every,
            "selection_steps": self.selection_step,
            "scored_observations": total_observations,
            "warmup_windows_after_switch": JOINT_RUNTIME_WARMUP_WINDOWS_AFTER_SWITCH,
            "throttle_detector": {
                "throughput_ratio_threshold": JOINT_RUNTIME_THROTTLE_THROUGHPUT_RATIO,
                "gpu_cost_ratio_threshold": JOINT_RUNTIME_THROTTLE_GPU_COST_RATIO,
                "requires_prior_confidence_windows": JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS,
            },
            "high_memory_pressure_bucket_threshold": JOINT_RUNTIME_HIGH_MEMORY_PRESSURE_BUCKET,
            "observation_decay": JOINT_RUNTIME_OBSERVATION_DECAY,
            "ucb_exploration_scale": JOINT_RUNTIME_UCB_EXPLORATION_SCALE,
            "winning_arm_index": self.best_measured_index(),
            "arms": self.arms.iter().zip(&self.measurements).enumerate().map(|(index, (arm, measurements))| {
                let lane_count = measurements
                    .devices
                    .iter()
                    .map(|device| device.lane_index)
                    .max()
                    .map(|lane_index| lane_index + 1)
                    .unwrap_or(0);
                let phase_service = serde_json::json!({
                    "adaptive_gradient_reduction_seconds": measurements.adaptive_gradient_reduction_service_seconds,
                    "adaptive_optimizer_boundary_seconds": measurements.adaptive_optimizer_boundary_service_seconds,
                });
                let window_quality = serde_json::json!({
                    "throttle_suspect_windows": measurements.throttle_suspect_windows,
                    "last_window_throttle_suspect": measurements.last_window_throttle_suspect,
                    "high_memory_pressure_windows": measurements.high_memory_pressure_windows,
                });
                let mut arm_report = serde_json::json!({
                    "index": index,
                    "gradient_stream_chunk_values": arm.gradient_stream_chunk_values,
                    "sequence_microbatch_size": arm.tape_geometry.sequence_microbatch_size,
                    "state_checkpoint_stride": arm.tape_geometry.state_checkpoint_stride,
                    "optimizer_broadcast_overlap": arm.optimizer_broadcast_overlap,
                    "windows": measurements.windows,
                    "tokens": measurements.tokens,
                    "elapsed_seconds": measurements.elapsed_seconds,
                    "tokens_per_second": measurements.tokens_per_second(),
                    "adaptive_tokens_per_second": measurements.adaptive_tokens_per_second,
                    "adaptive_effective_tokens_per_second": measurements.adaptive_effective_tokens_per_second,
                    "confidence_effective_samples": measurements.confidence_effective_samples,
                    "decayed_effective_confidence_samples": measurements.effective_confidence_samples(total_observations),
                    "last_observation_ordinal": measurements.last_observation_ordinal,
                    "observations_since_last_measurement": measurements.observations_since_last_measurement(total_observations),
                    "confidence_effective_mean_tokens_per_second": measurements.confidence_effective_mean,
                    "relative_uncertainty": measurements.relative_uncertainty_at(total_observations),
                    "confidence_adjusted_effective_tokens_per_second": measurements.confidence_adjusted_effective_tokens_per_second_at(total_observations),
                    "exploration_score_tokens_per_second": measurements.exploration_score_tokens_per_second_at(total_observations),
                    "queue_submissions": measurements.queue_submissions,
                    "queue_submissions_per_million_tokens": measurements.queue_submissions_per_million_tokens(),
                    "timeline_retirement_latency_samples": measurements.timeline_latency_samples,
                    "timeline_retirement_latency_ns_average": measurements.average_timeline_latency_ns(),
                    "adaptive_timeline_retirement_latency_ns": measurements.adaptive_timeline_latency_ns,
                    "kernel_profile_samples": measurements.kernel_profile_samples,
                    "kernel_dispatches": measurements.kernel_dispatches,
                    "kernel_gpu_ns_total": measurements.kernel_gpu_ns_total,
                    "kernel_gpu_ns_per_token": measurements.kernel_gpu_ns_per_token(),
                    "adaptive_kernel_gpu_ns_per_token": measurements.adaptive_kernel_gpu_ns_per_token,
                    "phase_service": phase_service,
                    "learned_device_workload_weights": self.lane_workload_weights(index, lane_count),
                    "devices": measurements
                        .devices
                        .iter()
                        .map(JointRuntimeDeviceMeasurements::report)
                        .collect::<Vec<_>>(),
                });
                if let Some(object) = arm_report.as_object_mut() {
                    object.insert("window_quality".to_owned(), window_quality);
                    object.insert(
                        "current_run_selected_windows".to_owned(),
                        serde_json::Value::from(self.current_run_selected_windows[index]),
                    );
                    object.insert(
                        "current_run_scored_windows".to_owned(),
                        serde_json::Value::from(self.current_run_scored_windows[index]),
                    );
                }
                arm_report
            }).collect::<Vec<_>>(),
        });
        if let Some(object) = report.as_object_mut() {
            object.insert("phase_selectors".to_owned(), phase_selectors);
            object.insert(
                "current_run_selected_windows".to_owned(),
                serde_json::Value::from(
                    self.current_run_selected_windows
                        .iter()
                        .copied()
                        .sum::<u64>(),
                ),
            );
            object.insert(
                "current_run_scored_windows".to_owned(),
                serde_json::Value::from(
                    self.current_run_scored_windows.iter().copied().sum::<u64>(),
                ),
            );
        }
        report
    }
}

struct ReplicaComputeJob {
    replica_index: usize,
    batch: usize,
    inputs: Vec<OwnedRawTokenLabeledSequenceInput>,
    objective: HierarchosLabeledSequenceObjective,
    hyper: AdamWHyperParams,
    policy: HierarchosTapeMemoryPolicy,
    readback: HierarchosTokenTapeReadbackPolicy,
    tape_geometry: Option<PhaseAwareTapeGeometry>,
    /// Host mirror of the primary window's current GradScaler state. Replica
    /// graphs never transition this state: they only use its scale to emit
    /// gradients in the same source-scaled domain as the primary. The primary
    /// remains the sole owner of overflow detection, scaler transition, and
    /// AdamW mutation after ordered reduction.
    loss_scaling: Option<HierarchosLossScalingState>,
    queued_at: Instant,
    shared_submission_arena: bool,
    collect_runtime_profile: bool,
    completion: mpsc::Sender<ReplicaComputeCompletion>,
}

struct ReplicaComputePayload {
    result: HierarchosBudgetedTokenTapeTrainResult,
    gradient_source: HierarchosPendingGradientTransportSource,
    runtime_profile: JointRuntimeDeviceWindowProfile,
}

struct ReplicaComputeCompletion {
    replica_index: usize,
    result: Result<ReplicaComputePayload>,
}

enum ReplicaWorkerJob {
    Broadcast(ReplicaStateBroadcastJob),
    Compute(ReplicaComputeJob),
}

struct ReplicaWorkerPool {
    senders: Vec<mpsc::Sender<ReplicaWorkerJob>>,
    retirement_timelines: Vec<Option<HierarchosReplicaStateRetirementTimeline>>,
    shared_primary_submission_arenas: Vec<bool>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl ReplicaWorkerPool {
    fn new(
        replicas: Vec<HierarchosTrainingGraph>,
        primary_device: &VulkanDevice,
        replica_devices: &[VulkanDevice],
        chunk_values: usize,
    ) -> Result<Self> {
        let replica_count = replicas.len();
        anyhow::ensure!(
            replica_devices.len() == replica_count,
            "replica worker pool received {} device views for {replica_count} graphs",
            replica_devices.len()
        );
        let mut senders = Vec::with_capacity(replica_count);
        let mut retirement_timelines = Vec::with_capacity(replica_count);
        let mut shared_primary_submission_arenas = Vec::with_capacity(replica_count);
        let mut workers = Vec::with_capacity(replica_count);
        for (replica_index, mut graph) in replicas.into_iter().enumerate() {
            let shared_primary_submission_arena =
                replica_devices[replica_index].shares_logical_device_with(primary_device);
            let step_metadata_words = graph.full_model_optimizer_step_metadata_words();
            let mut predeclared_transport = match HierarchosReplicaStateDeviceGroupTransport::new(
                primary_device,
                &replica_devices[replica_index],
                chunk_values,
                step_metadata_words,
            ) {
                Ok(transport) => transport,
                Err(err) => {
                    eprintln!(
                        "hierarchos_vulkan_replica_retirement_timeline_fallback replica={} source_device={} destination_device={} reason={err:#}",
                        replica_index,
                        primary_device.physical_device_index(),
                        replica_devices[replica_index].physical_device_index(),
                    );
                    None
                }
            };
            let retirement_timeline = predeclared_transport
                .as_ref()
                .map(HierarchosReplicaStateDeviceGroupTransport::retirement_timeline);
            let (sender, receiver) = mpsc::channel::<ReplicaWorkerJob>();
            let worker = std::thread::Builder::new()
                .name(format!("hierarchos-replica-worker-{replica_index}"))
                .spawn(move || {
                    let mut ready_for_compute = true;
                    while let Ok(job) = receiver.recv() {
                        match job {
                            ReplicaWorkerJob::Broadcast(job) => {
                                debug_assert_eq!(job.replica_index, replica_index);
                                let ReplicaStateBroadcastJob {
                                    replica_index,
                                    source,
                                    reset_accumulation,
                                    chunk_values,
                                    mut range_retirement,
                                    mut timeline_reservation,
                                    portable_state,
                                    completion,
                                } = job;
                                let result = (|| -> Result<ReplicaStateBroadcastSummary> {
                                    let mut summary = ReplicaStateBroadcastSummary::default();
                                    if reset_accumulation {
                                        graph.discard_full_model_accumulation_after_overflow()?;
                                        summary.queue_submissions = 1;
                                    }
                                    let streamed = if let Some(reservation) =
                                        timeline_reservation.as_mut()
                                    {
                                        let transport = predeclared_transport.as_mut().context(
                                            "replica broadcast received a predeclared timeline reservation without its persistent device-group transport",
                                        )?;
                                        Some(
                                            graph.stream_full_model_replica_state_from_source_with_predeclared_timeline(
                                                &source,
                                                chunk_values,
                                                transport,
                                                reservation,
                                            )?,
                                        )
                                    } else if let Some(retirement) = range_retirement.as_mut() {
                                        graph.stream_full_model_replica_state_from_source_with_range_retirement(
                                            &source,
                                            chunk_values,
                                            retirement,
                                        )?
                                    } else {
                                        graph.stream_full_model_replica_state_from_source(
                                            &source,
                                            chunk_values,
                                        )?
                                    };
                                    if let Some(stream) = streamed {
                                        absorb_replica_state_stream(&mut summary, stream)?;
                                    } else {
                                        let portable = portable_state.get_or_init(|| {
                                            source
                                                .portable_replica_state()
                                                .map(Arc::new)
                                                .map_err(|err| format!("{err:#}"))
                                        });
                                        let portable = portable.as_ref().map_err(|message| {
                                            anyhow::anyhow!(
                                                "materializing shared replica-state host fallback failed: {message}"
                                            )
                                        })?;
                                        graph.load_full_model_replica_state(portable)?;
                                        if let Some(retirement) = range_retirement.as_mut() {
                                            retirement.retire_all()?;
                                        }
                                        summary.host_fallbacks = 1;
                                    }
                                    Ok(summary)
                                })();
                                ready_for_compute = result.is_ok();
                                if completion
                                    .send(ReplicaStateBroadcastCompletion {
                                        replica_index,
                                        result,
                                    })
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            ReplicaWorkerJob::Compute(job) => {
                                debug_assert_eq!(job.replica_index, replica_index);
                                let ReplicaComputeJob {
                                    replica_index,
                                    batch,
                                    inputs,
                                    objective,
                                    hyper,
                                    policy,
                                    readback,
                                    tape_geometry,
                                    mut loss_scaling,
                                    queued_at,
                                    shared_submission_arena,
                                    collect_runtime_profile,
                                    completion,
                                } = job;
                                let result = (|| -> Result<ReplicaComputePayload> {
                                    anyhow::ensure!(
                                        ready_for_compute,
                                        "replica {replica_index} cannot compute because its preceding state broadcast failed or was not queued"
                                    );
                                    if let Some(transport) = predeclared_transport.as_mut() {
                                        // The preceding broadcast deliberately
                                        // returns before its replica queue is
                                        // idle. Reap only timeline-complete
                                        // owners here; the compute submissions
                                        // themselves remain correctly ordered
                                        // behind the async mirror refresh.
                                        transport.reap_completed_submissions()?;
                                    }
                                    let latency_before = collect_runtime_profile
                                        .then(|| graph.submission_arena_stats())
                                        .transpose()?;
                                    let borrowed = inputs
                                        .iter()
                                        .map(OwnedRawTokenLabeledSequenceInput::as_borrowed)
                                        .collect::<Vec<_>>();
                                    let result = if let Some(loss_scaling) = loss_scaling.as_mut() {
                                        train_zero_state_with_phase_aware_tape_geometry_dynamic(
                                            &mut graph,
                                            batch,
                                            &borrowed,
                                            objective,
                                            hyper,
                                            policy,
                                            readback,
                                            tape_geometry,
                                            loss_scaling,
                                            false,
                                        )?
                                    } else {
                                        train_zero_state_with_phase_aware_tape_geometry(
                                            &mut graph,
                                            batch,
                                            &borrowed,
                                            objective,
                                            hyper,
                                            policy,
                                            readback,
                                            tape_geometry,
                                        )?
                                    };
                                    let gradient_source = if loss_scaling.is_some() {
                                        graph.full_model_pending_gradient_transport_source_for_dynamic_loss_scaling()?
                                    } else {
                                        graph.full_model_pending_gradient_transport_source()?
                                    };
                                    let latency = if let Some(latency_before) = latency_before {
                                        SubmissionLatencyDelta::between(
                                            latency_before,
                                            graph.submission_arena_stats()?,
                                        )
                                    } else {
                                        SubmissionLatencyDelta::default()
                                    };
                                    let tokens_per_batch = inputs.iter().try_fold(0u64, |total, input| {
                                        total.checked_add(u64::try_from(input.tokens).context(
                                            "replica token count exceeds u64 for joint-runtime telemetry",
                                        )?)
                                        .context("replica joint-runtime token counter overflow")
                                    })?;
                                    let tokens = tokens_per_batch
                                        .checked_mul(u64::try_from(batch).context(
                                            "replica batch size exceeds u64 for joint-runtime telemetry",
                                        )?)
                                        .context("replica joint-runtime batch token counter overflow")?;
                                    let (
                                        device_local_usage_ratio,
                                        device_local_pressure_bucket,
                                    ) = joint_runtime_memory_pressure(
                                        &graph,
                                        collect_runtime_profile,
                                    )?;
                                    let runtime_profile = JointRuntimeDeviceWindowProfile {
                                        lane_index: replica_index + 1,
                                        tokens,
                                        elapsed_seconds: queued_at.elapsed().as_secs_f64(),
                                        queue_submissions: u64::from(result.queue_submissions),
                                        latency,
                                        shared_submission_arena,
                                        device_local_usage_ratio,
                                        device_local_pressure_bucket,
                                    };
                                    Ok(ReplicaComputePayload {
                                        result,
                                        gradient_source,
                                        runtime_profile,
                                    })
                                })();
                                // A successful compute leaves the replica's local
                                // accumulation open. It must consume the next
                                // optimizer-boundary broadcast before another
                                // compute job can begin.
                                ready_for_compute = result.is_err();
                                if completion
                                    .send(ReplicaComputeCompletion {
                                        replica_index,
                                        result,
                                    })
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                })
                .with_context(|| {
                    format!("spawning persistent replica execution worker {replica_index}")
                })?;
            senders.push(sender);
            retirement_timelines.push(retirement_timeline);
            shared_primary_submission_arenas.push(shared_primary_submission_arena);
            workers.push(worker);
        }
        Ok(Self {
            senders,
            retirement_timelines,
            shared_primary_submission_arenas,
            workers,
        })
    }

    fn launch(
        &self,
        source: HierarchosFullModelReplicaTransportSource,
        active_replica_count: usize,
        chunk_values: usize,
        enable_range_retirement: bool,
    ) -> Result<ReplicaStateBroadcastTicket> {
        let replica_count = self.senders.len();
        let mut range_retirements = if enable_range_retirement && replica_count != 0 {
            source.prepare_range_retirement_consumers(chunk_values, replica_count)?
        } else {
            Vec::new()
        };
        if enable_range_retirement {
            anyhow::ensure!(
                range_retirements.len() == replica_count,
                "replica range-retirement plan created {} consumers for {replica_count} workers",
                range_retirements.len()
            );
        }
        let source = Arc::new(source);
        let portable_state = Arc::new(OnceLock::new());
        let (completion, completed) = mpsc::channel();
        for (replica_index, sender) in self.senders.iter().enumerate() {
            let mut range_retirement = if enable_range_retirement {
                Some(range_retirements.remove(0))
            } else {
                None
            };
            let timeline_reservation = match (
                range_retirement.as_mut(),
                self.retirement_timelines
                    .get(replica_index)
                    .and_then(Option::as_ref),
            ) {
                (Some(retirement), Some(timeline)) => {
                    let reservation = timeline.reserve(&source, chunk_values)?;
                    retirement.predeclare_device_group_timeline(&reservation)?;
                    Some(reservation)
                }
                _ => None,
            };
            sender
                .send(ReplicaWorkerJob::Broadcast(ReplicaStateBroadcastJob {
                    replica_index,
                    source: Arc::clone(&source),
                    reset_accumulation: replica_index < active_replica_count,
                    chunk_values,
                    range_retirement,
                    timeline_reservation,
                    portable_state: Arc::clone(&portable_state),
                    completion: completion.clone(),
                }))
                .map_err(|_| {
                    anyhow::anyhow!(
                        "persistent replica-state broadcast worker {replica_index} disconnected"
                    )
                })?;
        }
        drop(completion);
        drop(source);
        Ok(ReplicaStateBroadcastTicket {
            replica_count,
            completed,
        })
    }

    fn predeclared_retirement_timeline_lanes(&self) -> usize {
        self.retirement_timelines.iter().flatten().count()
    }

    fn launch_compute(
        &self,
        batch: usize,
        replica_ranges: &[Range<usize>],
        inputs: &[HierarchosRawTokenLabeledSequenceInput<'_>],
        objective: HierarchosLabeledSequenceObjective,
        hyper: AdamWHyperParams,
        policy: HierarchosTapeMemoryPolicy,
        readback: HierarchosTokenTapeReadbackPolicy,
        tape_geometry: Option<PhaseAwareTapeGeometry>,
        loss_scaling: Option<&HierarchosLossScalingState>,
        collect_runtime_profile: bool,
    ) -> Result<ReplicaComputeTicket> {
        anyhow::ensure!(
            replica_ranges.len() <= self.senders.len(),
            "replica scheduler requested {} compute shards from {} persistent workers",
            replica_ranges.len(),
            self.senders.len()
        );
        let (completion, completed) = mpsc::channel();
        let queued_at = Instant::now();
        for (replica_index, range) in replica_ranges.iter().enumerate() {
            let owned_inputs = inputs[range.clone()]
                .iter()
                .map(OwnedRawTokenLabeledSequenceInput::capture)
                .collect::<Vec<_>>();
            self.senders[replica_index]
                .send(ReplicaWorkerJob::Compute(ReplicaComputeJob {
                    replica_index,
                    batch,
                    inputs: owned_inputs,
                    objective,
                    hyper,
                    policy,
                    readback,
                    tape_geometry,
                    loss_scaling: loss_scaling.cloned(),
                    queued_at,
                    shared_submission_arena: self
                        .shared_primary_submission_arenas
                        .get(replica_index)
                        .copied()
                        .unwrap_or(false),
                    collect_runtime_profile,
                    completion: completion.clone(),
                }))
                .map_err(|_| {
                    anyhow::anyhow!(
                        "persistent replica execution worker {replica_index} disconnected"
                    )
                })?;
        }
        drop(completion);
        Ok(ReplicaComputeTicket {
            replica_count: replica_ranges.len(),
            completed,
        })
    }
}

impl Drop for ReplicaWorkerPool {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

struct ReplicaStateBroadcastTicket {
    replica_count: usize,
    completed: mpsc::Receiver<ReplicaStateBroadcastCompletion>,
}

enum ReplicaStateBroadcastRetirement {
    Complete(ReplicaStateBroadcastSummary),
    Deferred(ReplicaStateBroadcastTicket),
}

impl ReplicaStateBroadcastRetirement {
    fn resolve(self) -> Result<ReplicaStateBroadcastSummary> {
        match self {
            Self::Complete(summary) => Ok(summary),
            Self::Deferred(ticket) => ticket.wait(),
        }
    }
}

impl ReplicaStateBroadcastTicket {
    fn recv_completion(&self) -> Result<ReplicaStateBroadcastCompletion> {
        self.completed
            .recv()
            .context("persistent replica-state broadcast worker disconnected")
    }

    fn wait(self) -> Result<ReplicaStateBroadcastSummary> {
        let mut seen = vec![false; self.replica_count];
        let mut summary = ReplicaStateBroadcastSummary::default();
        let mut first_error = None;
        for _ in 0..self.replica_count {
            let completion = self.recv_completion()?;
            anyhow::ensure!(
                completion.replica_index < self.replica_count,
                "replica-state worker returned invalid index {} for {} replicas",
                completion.replica_index,
                self.replica_count
            );
            anyhow::ensure!(
                !seen[completion.replica_index],
                "replica-state worker returned duplicate completion index {}",
                completion.replica_index
            );
            seen[completion.replica_index] = true;
            match completion.result {
                Ok(worker_summary) => {
                    merge_replica_state_broadcast_summary(&mut summary, worker_summary)?
                }
                Err(err) if first_error.is_none() => first_error = Some(err),
                Err(_) => {}
            }
        }
        if let Some(err) = first_error {
            return Err(err);
        }
        Ok(summary)
    }
}

struct ReplicaComputeTicket {
    replica_count: usize,
    completed: mpsc::Receiver<ReplicaComputeCompletion>,
}

impl ReplicaComputeTicket {
    fn recv_completion(&self) -> Result<ReplicaComputeCompletion> {
        self.completed
            .recv()
            .context("persistent replica execution worker disconnected")
    }
}

fn merge_replica_state_broadcast_summary(
    summary: &mut ReplicaStateBroadcastSummary,
    worker: ReplicaStateBroadcastSummary,
) -> Result<()> {
    summary.queue_submissions = summary
        .queue_submissions
        .checked_add(worker.queue_submissions)
        .context("replica-state worker submission counter overflow")?;
    summary.stream_chunks = summary
        .stream_chunks
        .checked_add(worker.stream_chunks)
        .context("replica-state worker chunk counter overflow")?;
    summary.stream_values = summary
        .stream_values
        .checked_add(worker.stream_values)
        .context("replica-state worker value counter overflow")?;
    summary.stream_pipeline_slots = summary
        .stream_pipeline_slots
        .max(worker.stream_pipeline_slots);
    summary.stream_persistent_reuses = summary
        .stream_persistent_reuses
        .checked_add(worker.stream_persistent_reuses)
        .context("replica-state worker persistent-reuse counter overflow")?;
    summary.stream_peak_host_bytes = summary
        .stream_peak_host_bytes
        .max(worker.stream_peak_host_bytes);
    summary.stream_peak_device_bytes = summary
        .stream_peak_device_bytes
        .max(worker.stream_peak_device_bytes);
    summary.stream_backends.extend(worker.stream_backends);
    summary.host_fallbacks = summary
        .host_fallbacks
        .checked_add(worker.host_fallbacks)
        .context("replica-state host-fallback counter overflow")?;
    Ok(())
}

fn absorb_replica_state_stream(
    summary: &mut ReplicaStateBroadcastSummary,
    stream: HierarchosReplicaStateStreamStats,
) -> Result<()> {
    summary.stream_backends.insert(stream.backend);
    summary.stream_chunks = summary
        .stream_chunks
        .checked_add(stream.chunk_count)
        .context("replica-state stream chunk counter overflow")?;
    summary.stream_values = summary
        .stream_values
        .checked_add(stream.value_count)
        .context("replica-state stream value counter overflow")?;
    summary.stream_pipeline_slots = summary.stream_pipeline_slots.max(stream.pipeline_slots);
    if stream.persistent_transport_reused {
        summary.stream_persistent_reuses = summary
            .stream_persistent_reuses
            .checked_add(1)
            .context("replica-state persistent transport reuse counter overflow")?;
    }
    summary.stream_peak_host_bytes = summary
        .stream_peak_host_bytes
        .max(stream.peak_host_state_bytes);
    summary.stream_peak_device_bytes = summary
        .stream_peak_device_bytes
        .max(stream.peak_device_state_bytes);
    summary.queue_submissions = summary
        .queue_submissions
        .checked_add(stream.queue_submissions as u64)
        .context("replica-state stream submission counter overflow")?;
    Ok(())
}

fn retire_replica_state_broadcast(
    pending_broadcast: &mut Option<ReplicaStateBroadcastTicket>,
    replica_count: usize,
) -> Result<ReplicaStateBroadcastSummary> {
    if let Some(ticket) = pending_broadcast.take() {
        anyhow::ensure!(
            ticket.replica_count == replica_count,
            "replica broadcast ticket owns {} workers; scheduler expects {replica_count}",
            ticket.replica_count
        );
        ticket.wait()
    } else {
        Ok(ReplicaStateBroadcastSummary::default())
    }
}

fn launch_replica_state_broadcast(
    primary: &HierarchosTrainingGraph,
    replica_pool: &ReplicaWorkerPool,
    pending_broadcast: &mut Option<ReplicaStateBroadcastTicket>,
    replica_count: usize,
    active_replica_count: usize,
    chunk_values: usize,
) -> Result<()> {
    anyhow::ensure!(
        pending_broadcast.is_none(),
        "cannot launch a replica broadcast while another ticket is in flight"
    );
    anyhow::ensure!(
        replica_pool.senders.len() == replica_count,
        "replica worker pool has {} lanes; expected {replica_count}",
        replica_pool.senders.len()
    );
    anyhow::ensure!(
        active_replica_count <= replica_count,
        "active replica count {active_replica_count} exceeds configured replicas {replica_count}"
    );
    if replica_count == 0 {
        return Ok(());
    }

    // Capture the optimizer-generation read lease before enqueueing the next
    // state transfer. Replica graphs remain resident in their long-lived worker
    // lanes; only this detached source crosses threads. The primary may begin a
    // new forward/backward window immediately, while AdamW mutation remains
    // guarded until every worker drops its source clone.
    let source = primary.full_model_replica_transport_source()?;
    // Every asynchronous broadcast is range-addressable. The next optimizer
    // boundary can therefore mutate canonical parameter/moment chunks as soon
    // as the slowest consumer of that individual chunk retires, without a
    // second model-sized optimizer generation.
    *pending_broadcast =
        Some(replica_pool.launch(source, active_replica_count, chunk_values, true)?);
    Ok(())
}

fn finish_full_model_accumulation_behind_replica_broadcast(
    primary: &mut HierarchosTrainingGraph,
    pending_broadcast: &mut Option<ReplicaStateBroadcastTicket>,
    replica_count: usize,
    hyper: AdamWHyperParams,
    grad_clip: f32,
    chunk_values: usize,
    optimizer_broadcast_overlap: bool,
    defer_broadcast_completion_join: bool,
) -> Result<(
    hierarchos_vulkan::RwkvOptimizerStepResult,
    usize,
    usize,
    ReplicaStateBroadcastRetirement,
)> {
    if optimizer_broadcast_overlap && pending_broadcast.is_some() {
        let (finish, wavefront_ranges) = primary
            .finish_full_model_accumulation_with_gradient_clipping_wavefront(
                hyper,
                grad_clip,
                chunk_values,
            )?;
        anyhow::ensure!(
            finish.stepped,
            "Vulkan optimizer window rejected non-finite gradients after normalization; global_norm={}",
            finish.total_norm
        );
        let retired_broadcast = if defer_broadcast_completion_join {
            let ticket = pending_broadcast
                .take()
                .context("optimizer wavefront lost its in-flight replica broadcast ticket")?;
            anyhow::ensure!(
                ticket.replica_count == replica_count,
                "replica broadcast ticket owns {} workers; scheduler expects {replica_count}",
                ticket.replica_count
            );
            ReplicaStateBroadcastRetirement::Deferred(ticket)
        } else {
            ReplicaStateBroadcastRetirement::Complete(retire_replica_state_broadcast(
                pending_broadcast,
                replica_count,
            )?)
        };
        Ok((
            finish.full_model_optimizer,
            wavefront_ranges,
            finish.queue_submissions as usize,
            retired_broadcast,
        ))
    } else {
        // The no-overlap arm deliberately inserts the old all-replica phase
        // boundary before AdamW. This is numerically identical to the wavefront
        // path and provides the autotuner with a real control arm for deciding
        // whether queue contention makes overlap counterproductive on a given
        // adapter topology.
        let retired_broadcast = retire_replica_state_broadcast(pending_broadcast, replica_count)?;
        let finish =
            primary.finish_full_model_accumulation_with_gradient_clipping(hyper, grad_clip)?;
        anyhow::ensure!(
            finish.stepped,
            "Vulkan optimizer window rejected non-finite gradients after normalization; global_norm={}",
            finish.total_norm
        );
        Ok((
            finish.full_model_optimizer,
            0,
            finish.queue_submissions as usize,
            ReplicaStateBroadcastRetirement::Complete(retired_broadcast),
        ))
    }
}

fn train_zero_state_with_phase_aware_tape_geometry(
    graph: &mut HierarchosTrainingGraph,
    batch: usize,
    inputs: &[HierarchosRawTokenLabeledSequenceInput<'_>],
    objective: HierarchosLabeledSequenceObjective,
    hyper: AdamWHyperParams,
    policy: HierarchosTapeMemoryPolicy,
    readback: HierarchosTokenTapeReadbackPolicy,
    tape_geometry: Option<PhaseAwareTapeGeometry>,
) -> Result<HierarchosBudgetedTokenTapeTrainResult> {
    let Some(tape_geometry) = tape_geometry else {
        return graph.train_zero_state_raw_token_labeled_sequences_budgeted_with_update_mode(
            batch,
            inputs,
            objective,
            hyper,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            policy,
            readback,
        );
    };
    anyhow::ensure!(
        !inputs.is_empty(),
        "phase-aware tape execution requires at least one sequence"
    );
    let requested_tokens_per_sequence = inputs
        .iter()
        .map(|input| input.tokens)
        .max()
        .context("phase-aware tape execution could not determine token width")?;
    anyhow::ensure!(
        requested_tokens_per_sequence > 0,
        "phase-aware tape execution does not accept empty sequences"
    );
    // Sharding can only make the preflight geometry smaller. Clamp the global
    // optimizer-window plan to the local shard so exact-plan validation remains
    // legal without increasing either tape residency or checkpoint density.
    let sequence_microbatch_size = tape_geometry
        .sequence_microbatch_size
        .min(inputs.len())
        .max(1);
    let state_checkpoint_stride = tape_geometry
        .state_checkpoint_stride
        .min(requested_tokens_per_sequence)
        .max(1);
    if let Err(err) = graph.plan_token_tape_memory_exact(
        batch,
        inputs.len(),
        requested_tokens_per_sequence,
        sequence_microbatch_size,
        state_checkpoint_stride,
        policy,
    ) {
        eprintln!(
            "vulkan_phase_tape_runtime_revalidation fallback=adaptive-tape-planner sequences={} tokens={} planned_microbatch={} planned_stride={} reason={err:#}",
            inputs.len(),
            requested_tokens_per_sequence,
            sequence_microbatch_size,
            state_checkpoint_stride,
        );
        return graph.train_zero_state_raw_token_labeled_sequences_budgeted_with_update_mode(
            batch,
            inputs,
            objective,
            hyper,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            policy,
            readback,
        );
    }
    graph.train_zero_state_raw_token_labeled_sequences_with_plan_and_update_mode(
        batch,
        inputs,
        objective,
        hyper,
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
        sequence_microbatch_size,
        state_checkpoint_stride,
        policy,
        readback,
    )
}

#[allow(clippy::too_many_arguments)]
fn train_zero_state_with_phase_aware_tape_geometry_dynamic(
    graph: &mut HierarchosTrainingGraph,
    batch: usize,
    inputs: &[HierarchosRawTokenLabeledSequenceInput<'_>],
    objective: HierarchosLabeledSequenceObjective,
    hyper: AdamWHyperParams,
    policy: HierarchosTapeMemoryPolicy,
    readback: HierarchosTokenTapeReadbackPolicy,
    tape_geometry: Option<PhaseAwareTapeGeometry>,
    loss_scaling: &mut HierarchosLossScalingState,
    device_resident_backward_scale: bool,
) -> Result<HierarchosBudgetedTokenTapeTrainResult> {
    let Some(tape_geometry) = tape_geometry else {
        return if device_resident_backward_scale {
            graph.train_zero_state_raw_token_labeled_sequences_budgeted_with_device_resident_dynamic_loss_scaling(
                batch,
                inputs,
                objective,
                hyper,
                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                policy,
                readback,
                loss_scaling,
            )
        } else {
            graph.train_zero_state_raw_token_labeled_sequences_budgeted_with_dynamic_loss_scaling(
                batch,
                inputs,
                objective,
                hyper,
                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                policy,
                readback,
                loss_scaling,
            )
        };
    };
    anyhow::ensure!(
        !inputs.is_empty(),
        "phase-aware dynamic tape execution requires at least one sequence"
    );
    let requested_tokens_per_sequence = inputs
        .iter()
        .map(|input| input.tokens)
        .max()
        .context("phase-aware dynamic tape execution could not determine token width")?;
    anyhow::ensure!(
        requested_tokens_per_sequence > 0,
        "phase-aware dynamic tape execution does not accept empty sequences"
    );
    let sequence_microbatch_size = tape_geometry
        .sequence_microbatch_size
        .min(inputs.len())
        .max(1);
    let state_checkpoint_stride = tape_geometry
        .state_checkpoint_stride
        .min(requested_tokens_per_sequence)
        .max(1);
    if let Err(err) = graph.plan_token_tape_memory_exact(
        batch,
        inputs.len(),
        requested_tokens_per_sequence,
        sequence_microbatch_size,
        state_checkpoint_stride,
        policy,
    ) {
        eprintln!(
            "vulkan_phase_tape_runtime_revalidation mode=dynamic fallback=adaptive-tape-planner sequences={} tokens={} planned_microbatch={} planned_stride={} reason={err:#}",
            inputs.len(),
            requested_tokens_per_sequence,
            sequence_microbatch_size,
            state_checkpoint_stride,
        );
        return if device_resident_backward_scale {
            graph.train_zero_state_raw_token_labeled_sequences_budgeted_with_device_resident_dynamic_loss_scaling(
                batch,
                inputs,
                objective,
                hyper,
                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                policy,
                readback,
                loss_scaling,
            )
        } else {
            graph.train_zero_state_raw_token_labeled_sequences_budgeted_with_dynamic_loss_scaling(
                batch,
                inputs,
                objective,
                hyper,
                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                policy,
                readback,
                loss_scaling,
            )
        };
    }
    if device_resident_backward_scale {
        graph.train_zero_state_raw_token_labeled_sequences_with_plan_and_device_resident_dynamic_loss_scaling(
            batch,
            inputs,
            objective,
            hyper,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            sequence_microbatch_size,
            state_checkpoint_stride,
            policy,
            readback,
            loss_scaling,
        )
    } else {
        graph.train_zero_state_raw_token_labeled_sequences_with_plan_and_dynamic_loss_scaling(
            batch,
            inputs,
            objective,
            hyper,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            sequence_microbatch_size,
            state_checkpoint_stride,
            policy,
            readback,
            loss_scaling,
        )
    }
}

struct DynamicBudgetedWindow<'a> {
    loss_scaling: &'a mut HierarchosLossScalingState,
    device_resident_backward_scale: bool,
    optimizer_step_before: u32,
}

/// Close one globally reduced AMP window while preserving the same replica-
/// broadcast overlap contract as the ordinary FP32 scheduler. The primary is
/// the only graph that advances GradScaler and AdamW. Replica gradients merely
/// arrive in the same scaled domain and may contain NaN/Inf; the queue-resident
/// primary finisher owns the one PyTorch-style overflow decision.
#[allow(clippy::too_many_arguments)]
fn finish_dynamic_full_model_accumulation_behind_replica_broadcast(
    primary: &mut HierarchosTrainingGraph,
    pending_broadcast: &mut Option<ReplicaStateBroadcastTicket>,
    replica_count: usize,
    hyper: AdamWHyperParams,
    loss_scaling: &mut HierarchosLossScalingState,
    grad_clip: f32,
    chunk_values: usize,
    optimizer_broadcast_overlap: bool,
    defer_broadcast_completion_join: bool,
    optimizer_step_before: u32,
) -> Result<(u32, usize, usize, ReplicaStateBroadcastRetirement)> {
    // The control arm retires the previous replica generation before entering
    // AdamW. The overlap arm instead lets the optimizer's generation guard
    // consume predeclared per-range timeline retirement while it advances.
    let retired_before_finish = if optimizer_broadcast_overlap {
        None
    } else {
        Some(retire_replica_state_broadcast(
            pending_broadcast,
            replica_count,
        )?)
    };

    let finish = primary
        .finish_full_model_accumulation_with_dynamic_loss_scaling_and_gradient_clipping_wavefront_device_resident(
            hyper,
            loss_scaling,
            grad_clip,
            chunk_values,
        )?;
    let (decision, observation_submissions) = primary
        .observe_dynamic_loss_scaling_decision_preserving_device_authority_with_submission_count(
        )?;

    // Keep a tiny scheduler shadow of the device-owned scaler. This does not
    // transfer authority back to Rust: the next primary backward still reads
    // Vulkan controller word zero directly. The shadow exists so independent
    // logical devices can receive the exact same source scale for their next
    // data-parallel shard without owning or transitioning a second GradScaler.
    loss_scaling.scale = Some(decision.scale_after);
    loss_scaling.growth_tracker = Some(decision.growth_tracker);
    loss_scaling.pending_gradients_scaled = false;

    let optimizer_step = if decision.should_step {
        optimizer_step_before
            .checked_add(1)
            .context("native queue-resident dynamic optimizer-step counter overflow")?
    } else {
        optimizer_step_before
    };
    let retired_broadcast = if let Some(summary) = retired_before_finish {
        ReplicaStateBroadcastRetirement::Complete(summary)
    } else if defer_broadcast_completion_join && pending_broadcast.is_some() {
        let ticket = pending_broadcast
            .take()
            .context("dynamic optimizer wavefront lost its in-flight replica broadcast ticket")?;
        anyhow::ensure!(
            ticket.replica_count == replica_count,
            "replica broadcast ticket owns {} workers; scheduler expects {replica_count}",
            ticket.replica_count
        );
        ReplicaStateBroadcastRetirement::Deferred(ticket)
    } else {
        ReplicaStateBroadcastRetirement::Complete(retire_replica_state_broadcast(
            pending_broadcast,
            replica_count,
        )?)
    };

    // The scheduler still mirrors the tiny GradScaler decision once per global
    // window because replica source scaling and host LR/error-budget clocks
    // require it. Device-resident finishers piggyback that telemetry copy onto
    // their preparation submission, so this observation normally contributes
    // no additional Vulkan submission.
    let observation_submissions = usize::try_from(observation_submissions)
        .context("dynamic optimizer observation submission count exceeds usize")?;
    let queue_submissions = usize::try_from(finish.queue_submissions)
        .context("dynamic optimizer submission count exceeds usize")?
        .checked_add(observation_submissions)
        .context("dynamic optimizer submission count overflow")?;
    Ok((
        optimizer_step,
        finish.state_ranges,
        queue_submissions,
        retired_broadcast,
    ))
}

fn train_budgeted_window(
    primary: &mut HierarchosTrainingGraph,
    replica_pool: &ReplicaWorkerPool,
    pending_broadcast: &mut Option<ReplicaStateBroadcastTicket>,
    replica_count: usize,
    batch: usize,
    inputs: &[HierarchosRawTokenLabeledSequenceInput<'_>],
    objective: HierarchosLabeledSequenceObjective,
    hyper: AdamWHyperParams,
    grad_clip: f32,
    gradient_stream_chunk_values: usize,
    tape_geometry: Option<PhaseAwareTapeGeometry>,
    optimizer_broadcast_overlap: bool,
    lane_workload_weights: Option<&[f64]>,
    collect_runtime_profile: bool,
    mut dynamic: Option<DynamicBudgetedWindow<'_>>,
) -> Result<BudgetedWindowTrainSummary> {
    anyhow::ensure!(!inputs.is_empty(), "budgeted training window is empty");
    anyhow::ensure!(
        replica_pool.senders.len() == replica_count,
        "replica worker pool has {} lanes; scheduler expects {replica_count}",
        replica_pool.senders.len()
    );
    let policy = HierarchosTapeMemoryPolicy::default();
    let readback = HierarchosTokenTapeReadbackPolicy::LossOnly;
    if let Some(dynamic) = dynamic.as_ref() {
        anyhow::ensure!(
            !dynamic.loss_scaling.pending_gradients_scaled,
            "dynamic data-parallel window cannot begin with pending scaled gradients"
        );
    }
    // Replica graphs receive a per-window clone only as a backward-source
    // scalar carrier. They never resolve overflow or mutate this state.
    let replica_loss_scaling = dynamic
        .as_ref()
        .map(|dynamic| (*dynamic.loss_scaling).clone());
    let token_weights = inputs.iter().map(|input| input.tokens).collect::<Vec<_>>();
    let ranges = if let Some(lane_workload_weights) = lane_workload_weights {
        anyhow::ensure!(
            lane_workload_weights.len() == replica_count + 1,
            "joint runtime scheduler provided {} lane workload weights for {} execution lanes",
            lane_workload_weights.len(),
            replica_count + 1
        );
        contiguous_weighted_shard_ranges_by_capacity(&token_weights, lane_workload_weights)?
    } else {
        contiguous_weighted_shard_ranges(&token_weights, replica_count + 1)?
    };

    if ranges.len() == 1 {
        // This is the important generation-overlap seam even when a window is
        // too small to occupy a replica. The previous optimizer boundary may
        // still be draining through replica workers while primary forward /
        // backward advances against the immutable captured generation.
        let primary_latency_before = collect_runtime_profile
            .then(|| primary.submission_arena_stats())
            .transpose()?;
        let primary_started = Instant::now();
        let primary_tokens =
            inputs[ranges[0].clone()]
                .iter()
                .try_fold(0u64, |total, input| {
                    total
                        .checked_add(u64::try_from(input.tokens).context(
                            "primary token count exceeds u64 for joint-runtime telemetry",
                        )?)
                        .context("primary joint-runtime token counter overflow")
                })?;
        let primary_tokens = primary_tokens
            .checked_mul(
                u64::try_from(batch)
                    .context("primary batch size exceeds u64 for joint-runtime telemetry")?,
            )
            .context("primary joint-runtime batch token counter overflow")?;
        let result = if let Some(dynamic) = dynamic.as_mut() {
            train_zero_state_with_phase_aware_tape_geometry_dynamic(
                primary,
                batch,
                &inputs[ranges[0].clone()],
                objective,
                hyper,
                policy,
                readback,
                tape_geometry,
                dynamic.loss_scaling,
                dynamic.device_resident_backward_scale,
            )?
        } else {
            train_zero_state_with_phase_aware_tape_geometry(
                primary,
                batch,
                &inputs[ranges[0].clone()],
                objective,
                hyper,
                policy,
                readback,
                tape_geometry,
            )?
        };
        let primary_latency = if let Some(primary_latency_before) = primary_latency_before {
            SubmissionLatencyDelta::between(
                primary_latency_before,
                primary.submission_arena_stats()?,
            )
        } else {
            SubmissionLatencyDelta::default()
        };
        let (device_local_usage_ratio, device_local_pressure_bucket) =
            joint_runtime_memory_pressure(primary, collect_runtime_profile)?;
        let runtime_device_profiles = vec![JointRuntimeDeviceWindowProfile {
            lane_index: 0,
            tokens: primary_tokens,
            elapsed_seconds: primary_started.elapsed().as_secs_f64(),
            queue_submissions: u64::from(result.queue_submissions),
            latency: primary_latency,
            shared_submission_arena: false,
            device_local_usage_ratio,
            device_local_pressure_bucket,
        }];
        // If the previous generation is still broadcasting, AdamW now follows
        // its per-range retirement wavefront instead of waiting for the whole
        // ticket. The ticket is collected afterward only for error propagation
        // and transport telemetry.
        let optimizer_boundary_started = Instant::now();
        let defer_broadcast_completion_join = optimizer_broadcast_overlap
            && replica_pool.predeclared_retirement_timeline_lanes() == replica_count;
        let (
            optimizer_step,
            optimizer_wavefront_ranges,
            optimizer_submissions,
            broadcast_retirement,
        ) = if let Some(dynamic) = dynamic.as_mut() {
            finish_dynamic_full_model_accumulation_behind_replica_broadcast(
                primary,
                pending_broadcast,
                replica_count,
                hyper,
                dynamic.loss_scaling,
                grad_clip,
                gradient_stream_chunk_values,
                optimizer_broadcast_overlap,
                defer_broadcast_completion_join,
                dynamic.optimizer_step_before,
            )?
        } else {
            let (optimizer, wavefront_ranges, submissions, retirement) =
                finish_full_model_accumulation_behind_replica_broadcast(
                    primary,
                    pending_broadcast,
                    replica_count,
                    hyper,
                    grad_clip,
                    gradient_stream_chunk_values,
                    optimizer_broadcast_overlap,
                    defer_broadcast_completion_join,
                )?;
            (optimizer.step, wavefront_ranges, submissions, retirement)
        };
        let runtime_phase_profile = JointRuntimePhaseWindowProfile {
            gradient_reduction_service_seconds: 0.0,
            optimizer_boundary_service_seconds: optimizer_boundary_started.elapsed().as_secs_f64(),
        };
        let optimizer_wavefront_windows = usize::from(optimizer_wavefront_ranges != 0);
        let losses = result
            .sequences
            .into_iter()
            .flat_map(|sequence| sequence.losses)
            .collect::<Vec<_>>();
        launch_replica_state_broadcast(
            primary,
            replica_pool,
            pending_broadcast,
            replica_count,
            0,
            gradient_stream_chunk_values,
        )?;
        // The next generation is now in the worker queues before the CPU joins
        // the previous generation's completion channel. Timeline-capable
        // workers will carry the optimizer tail dependency into their first
        // source copy, so this join is error/telemetry collection rather than a
        // replica-generation scheduling barrier.
        let retired_broadcast = broadcast_retirement.resolve()?;
        let queue_submissions = u64::from(result.queue_submissions)
            .checked_add(optimizer_submissions as u64)
            .and_then(|count| count.checked_add(retired_broadcast.queue_submissions))
            .context("single-shard pipelined submission count overflow")?;
        return Ok(BudgetedWindowTrainSummary {
            optimizer_step,
            queue_submissions,
            optimizer_wavefront_windows,
            optimizer_wavefront_ranges,
            losses,
            gradient_reductions: 0,
            replica_broadcasts: replica_count,
            replica_broadcast_compute_handoffs: 0,
            gradient_stream_chunks: 0,
            gradient_stream_values: 0,
            gradient_stream_pipeline_slots: 0,
            gradient_stream_persistent_reuses: 0,
            gradient_stream_peak_host_bytes: 0,
            gradient_stream_peak_device_bytes: 0,
            gradient_stream_peak_host_heap_bytes: 0,
            gradient_stream_backends: HashSet::new(),
            replica_state_stream_chunks: retired_broadcast.stream_chunks,
            replica_state_stream_values: retired_broadcast.stream_values,
            replica_state_stream_pipeline_slots: retired_broadcast.stream_pipeline_slots,
            replica_state_stream_persistent_reuses: retired_broadcast.stream_persistent_reuses,
            replica_state_stream_peak_host_bytes: retired_broadcast.stream_peak_host_bytes,
            replica_state_stream_peak_device_bytes: retired_broadcast.stream_peak_device_bytes,
            replica_state_stream_backends: retired_broadcast.stream_backends,
            replica_state_host_fallbacks: retired_broadcast.host_fallbacks,
            runtime_device_profiles,
            runtime_phase_profile,
        });
    }

    let primary_range = ranges[0].clone();
    let replica_ranges = &ranges[1..];
    let window_replica_count = replica_ranges.len();
    anyhow::ensure!(
        window_replica_count <= replica_count,
        "budgeted window needs {window_replica_count} replica shards but only {replica_count} workers exist"
    );

    // Queue replica work before primary compute. mpsc FIFO gives each resident
    // worker a tiny per-device command stream: [old broadcast] -> [new compute].
    // Thus a replica starts the next window as soon as *its* state transfer
    // retires, without waiting for slower peers or transferring graph ownership.
    let replica_broadcast_compute_handoffs = if pending_broadcast.is_some() {
        window_replica_count
    } else {
        0
    };
    let primary_latency_before = collect_runtime_profile
        .then(|| primary.submission_arena_stats())
        .transpose()?;
    let primary_started = Instant::now();
    let primary_tokens = inputs[primary_range.clone()]
        .iter()
        .try_fold(0u64, |total, input| {
            total
                .checked_add(
                    u64::try_from(input.tokens)
                        .context("primary token count exceeds u64 for joint-runtime telemetry")?,
                )
                .context("primary joint-runtime token counter overflow")
        })?
        .checked_mul(
            u64::try_from(batch)
                .context("primary batch size exceeds u64 for joint-runtime telemetry")?,
        )
        .context("primary joint-runtime batch token counter overflow")?;
    let replica_ticket = replica_pool.launch_compute(
        batch,
        replica_ranges,
        inputs,
        objective,
        hyper,
        policy,
        readback,
        tape_geometry,
        replica_loss_scaling.as_ref(),
        collect_runtime_profile,
    )?;
    anyhow::ensure!(
        replica_ticket.replica_count == window_replica_count,
        "replica compute ticket owns {} shards; scheduler expects {window_replica_count}",
        replica_ticket.replica_count
    );

    // Primary forward/backward is deliberately allowed to run while the prior
    // generation is still being drained to replicas. No AdamW mutation occurs
    // here, so the optimizer-generation read lease remains sufficient.
    let primary_result = if let Some(dynamic) = dynamic.as_mut() {
        train_zero_state_with_phase_aware_tape_geometry_dynamic(
            primary,
            batch,
            &inputs[primary_range],
            objective,
            hyper,
            policy,
            readback,
            tape_geometry,
            dynamic.loss_scaling,
            dynamic.device_resident_backward_scale,
        )?
    } else {
        train_zero_state_with_phase_aware_tape_geometry(
            primary,
            batch,
            &inputs[primary_range],
            objective,
            hyper,
            policy,
            readback,
            tape_geometry,
        )?
    };
    let primary_latency = if let Some(primary_latency_before) = primary_latency_before {
        SubmissionLatencyDelta::between(primary_latency_before, primary.submission_arena_stats()?)
    } else {
        SubmissionLatencyDelta::default()
    };
    let (device_local_usage_ratio, device_local_pressure_bucket) =
        joint_runtime_memory_pressure(primary, collect_runtime_profile)?;

    let mut queue_submissions = u64::from(primary_result.queue_submissions);
    let mut runtime_device_profiles = vec![JointRuntimeDeviceWindowProfile {
        lane_index: 0,
        tokens: primary_tokens,
        elapsed_seconds: primary_started.elapsed().as_secs_f64(),
        queue_submissions: u64::from(primary_result.queue_submissions),
        latency: primary_latency,
        shared_submission_arena: replica_pool
            .shared_primary_submission_arenas
            .iter()
            .take(window_replica_count)
            .any(|shared| *shared),
        device_local_usage_ratio,
        device_local_pressure_bucket,
    }];
    let mut losses = primary_result
        .sequences
        .into_iter()
        .flat_map(|sequence| sequence.losses)
        .collect::<Vec<_>>();
    let mut gradient_stream_chunks = 0usize;
    let mut gradient_stream_values = 0usize;
    let mut gradient_stream_pipeline_slots = 0usize;
    let mut gradient_stream_persistent_reuses = 0usize;
    let mut gradient_stream_peak_host_bytes = 0usize;
    let mut gradient_stream_peak_device_bytes = 0usize;
    let mut gradient_stream_peak_host_heap_bytes = 0usize;
    let mut gradient_stream_backends = HashSet::new();
    let mut gradient_reduction_service_seconds = 0.0f64;

    // Completions may arrive in any order, but gradient reduction must not.
    // Buffer completed payloads until the next canonical replica index is ready;
    // then reduce immediately. This preserves deterministic accumulation order
    // while letting later replica backward / DMA continue in parallel.
    let mut completed_payloads = (0..window_replica_count)
        .map(|_| None)
        .collect::<Vec<Option<ReplicaComputePayload>>>();
    let mut seen = vec![false; window_replica_count];
    let mut next_replica = 0usize;
    for _ in 0..window_replica_count {
        let completion = replica_ticket.recv_completion()?;
        anyhow::ensure!(
            completion.replica_index < window_replica_count,
            "replica compute worker returned invalid index {} for {window_replica_count} active replicas",
            completion.replica_index
        );
        anyhow::ensure!(
            !seen[completion.replica_index],
            "replica compute worker returned duplicate completion index {}",
            completion.replica_index
        );
        seen[completion.replica_index] = true;
        let replica_index = completion.replica_index;
        completed_payloads[replica_index] = Some(completion.result.with_context(|| {
            format!("persistent replica execution worker {replica_index} failed")
        })?);

        while next_replica < window_replica_count && completed_payloads[next_replica].is_some() {
            let ReplicaComputePayload {
                result,
                gradient_source,
                runtime_profile,
            } = completed_payloads[next_replica]
                .take()
                .context("ordered replica reduction payload disappeared")?;
            runtime_device_profiles.push(runtime_profile);

            queue_submissions = queue_submissions
                .checked_add(u64::from(result.queue_submissions))
                .context("multi-device queue-submission count overflow")?;
            losses.extend(
                result
                    .sequences
                    .into_iter()
                    .flat_map(|sequence| sequence.losses),
            );

            let gradient_reduction_started = Instant::now();
            let stream = primary.accumulate_full_model_pending_gradients_streamed_from_source(
                &gradient_source,
                gradient_stream_chunk_values,
            )?;
            gradient_reduction_service_seconds +=
                gradient_reduction_started.elapsed().as_secs_f64();
            gradient_stream_backends.insert(stream.backend);
            gradient_stream_chunks = gradient_stream_chunks
                .checked_add(stream.chunk_count)
                .context("multi-device gradient-stream chunk counter overflow")?;
            gradient_stream_values = gradient_stream_values
                .checked_add(stream.value_count)
                .context("multi-device gradient-stream value counter overflow")?;
            gradient_stream_pipeline_slots =
                gradient_stream_pipeline_slots.max(stream.pipeline_slots);
            if stream.persistent_transport_reused {
                gradient_stream_persistent_reuses = gradient_stream_persistent_reuses
                    .checked_add(1)
                    .context("multi-device persistent transport reuse counter overflow")?;
            }
            gradient_stream_peak_host_bytes =
                gradient_stream_peak_host_bytes.max(stream.peak_host_gradient_bytes);
            gradient_stream_peak_device_bytes =
                gradient_stream_peak_device_bytes.max(stream.peak_device_gradient_bytes);
            gradient_stream_peak_host_heap_bytes =
                gradient_stream_peak_host_heap_bytes.max(stream.peak_host_heap_gradient_bytes);
            queue_submissions = queue_submissions
                .checked_add(stream.queue_submissions as u64)
                .context("multi-device reduction submission count overflow")?;
            next_replica = next_replica
                .checked_add(1)
                .context("gradient reduction replica-order counter overflow")?;
        }
    }
    anyhow::ensure!(
        next_replica == window_replica_count,
        "ordered gradient reducer retired {next_replica} replicas; expected {window_replica_count}"
    );

    // Active replicas have completed their old broadcast before entering
    // compute, but idle replicas may still be draining it. AdamW can now trail
    // those remaining DMA fences chunk-by-chunk instead of introducing a final
    // all-replica phase barrier.
    let optimizer_boundary_started = Instant::now();
    let defer_broadcast_completion_join = optimizer_broadcast_overlap
        && replica_pool.predeclared_retirement_timeline_lanes() == replica_count;
    let (optimizer_step, optimizer_wavefront_ranges, optimizer_submissions, broadcast_retirement) =
        if let Some(dynamic) = dynamic.as_mut() {
            finish_dynamic_full_model_accumulation_behind_replica_broadcast(
                primary,
                pending_broadcast,
                replica_count,
                hyper,
                dynamic.loss_scaling,
                grad_clip,
                gradient_stream_chunk_values,
                optimizer_broadcast_overlap,
                defer_broadcast_completion_join,
                dynamic.optimizer_step_before,
            )?
        } else {
            let (optimizer, wavefront_ranges, submissions, retirement) =
                finish_full_model_accumulation_behind_replica_broadcast(
                    primary,
                    pending_broadcast,
                    replica_count,
                    hyper,
                    grad_clip,
                    gradient_stream_chunk_values,
                    optimizer_broadcast_overlap,
                    defer_broadcast_completion_join,
                )?;
            (optimizer.step, wavefront_ranges, submissions, retirement)
        };
    let runtime_phase_profile = JointRuntimePhaseWindowProfile {
        gradient_reduction_service_seconds,
        optimizer_boundary_service_seconds: optimizer_boundary_started.elapsed().as_secs_f64(),
    };
    let optimizer_wavefront_windows = usize::from(optimizer_wavefront_ranges != 0);
    queue_submissions = queue_submissions
        .checked_add(optimizer_submissions as u64)
        .context("multi-device optimizer wavefront submission count overflow")?;

    launch_replica_state_broadcast(
        primary,
        replica_pool,
        pending_broadcast,
        replica_count,
        window_replica_count,
        gradient_stream_chunk_values,
    )?;
    // Queue the next state generation first. The overlap arm can therefore
    // leave the previous completion channel entirely off the scheduling path;
    // resolving it here preserves exact worker failures and transport telemetry.
    // The no-overlap control arm already resolved its ticket before AdamW.
    let retired_broadcast = broadcast_retirement.resolve()?;
    queue_submissions = queue_submissions
        .checked_add(retired_broadcast.queue_submissions)
        .context("multi-device retired broadcast submission count overflow")?;
    runtime_device_profiles.sort_by_key(|profile| profile.lane_index);

    Ok(BudgetedWindowTrainSummary {
        optimizer_step,
        queue_submissions,
        optimizer_wavefront_windows,
        optimizer_wavefront_ranges,
        losses,
        gradient_reductions: window_replica_count,
        replica_broadcasts: replica_count,
        replica_broadcast_compute_handoffs,
        gradient_stream_chunks,
        gradient_stream_values,
        gradient_stream_pipeline_slots,
        gradient_stream_persistent_reuses,
        gradient_stream_peak_host_bytes,
        gradient_stream_peak_device_bytes,
        gradient_stream_peak_host_heap_bytes,
        gradient_stream_backends,
        replica_state_stream_chunks: retired_broadcast.stream_chunks,
        replica_state_stream_values: retired_broadcast.stream_values,
        replica_state_stream_pipeline_slots: retired_broadcast.stream_pipeline_slots,
        replica_state_stream_persistent_reuses: retired_broadcast.stream_persistent_reuses,
        replica_state_stream_peak_host_bytes: retired_broadcast.stream_peak_host_bytes,
        replica_state_stream_peak_device_bytes: retired_broadcast.stream_peak_device_bytes,
        replica_state_stream_backends: retired_broadcast.stream_backends,
        replica_state_host_fallbacks: retired_broadcast.host_fallbacks,
        runtime_device_profiles,
        runtime_phase_profile,
    })
}

#[cfg(any())]
fn train_budgeted_window_scoped_legacy(
    primary: &mut HierarchosTrainingGraph,
    broadcast_pool: &ReplicaStateBroadcastPool,
    replica_graphs: &mut Option<Vec<HierarchosTrainingGraph>>,
    pending_broadcast: &mut Option<ReplicaStateBroadcastTicket>,
    replica_count: usize,
    batch: usize,
    inputs: &[HierarchosRawTokenLabeledSequenceInput<'_>],
    objective: HierarchosLabeledSequenceObjective,
    hyper: AdamWHyperParams,
    gradient_stream_chunk_values: usize,
) -> Result<BudgetedWindowTrainSummary> {
    anyhow::ensure!(!inputs.is_empty(), "budgeted training window is empty");
    let policy = HierarchosTapeMemoryPolicy::default();
    let readback = HierarchosTokenTapeReadbackPolicy::LossOnly;
    let token_weights = inputs.iter().map(|input| input.tokens).collect::<Vec<_>>();
    let ranges = contiguous_weighted_shard_ranges(&token_weights, replica_count + 1)?;

    if ranges.len() == 1 {
        let result = primary
            .train_zero_state_raw_token_labeled_sequences_budgeted_with_update_mode(
                batch,
                &inputs[ranges[0].clone()],
                objective,
                hyper,
                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                policy,
                readback,
            )?;
        // The primary forward/backward submission above is allowed to overlap
        // an older broadcast ticket. Recover replicas only after that work has
        // been launched/completed, then cross the guarded AdamW boundary.
        let retired_broadcast =
            retire_replica_state_broadcast(replica_graphs, pending_broadcast, replica_count)?;
        let optimizer = primary.finish_full_model_accumulation(hyper)?;
        let queue_submissions = u64::from(result.queue_submissions)
            .checked_add(1)
            .and_then(|count| count.checked_add(retired_broadcast.queue_submissions))
            .context("single-shard pipelined submission count overflow")?;
        let losses = result
            .sequences
            .into_iter()
            .flat_map(|sequence| sequence.losses)
            .collect::<Vec<_>>();
        launch_replica_state_broadcast(
            primary,
            broadcast_pool,
            replica_graphs,
            pending_broadcast,
            replica_count,
            0,
            gradient_stream_chunk_values,
        )?;
        return Ok(BudgetedWindowTrainSummary {
            optimizer_step: optimizer.step,
            queue_submissions,
            losses,
            gradient_reductions: 0,
            replica_broadcasts: replica_count,
            replica_broadcast_compute_handoffs: 0,
            gradient_stream_chunks: 0,
            gradient_stream_values: 0,
            gradient_stream_pipeline_slots: 0,
            gradient_stream_persistent_reuses: 0,
            gradient_stream_peak_host_bytes: 0,
            gradient_stream_peak_device_bytes: 0,
            gradient_stream_peak_host_heap_bytes: 0,
            gradient_stream_backends: HashSet::new(),
            replica_state_stream_chunks: retired_broadcast.stream_chunks,
            replica_state_stream_values: retired_broadcast.stream_values,
            replica_state_stream_pipeline_slots: retired_broadcast.stream_pipeline_slots,
            replica_state_stream_persistent_reuses: retired_broadcast.stream_persistent_reuses,
            replica_state_stream_peak_host_bytes: retired_broadcast.stream_peak_host_bytes,
            replica_state_stream_peak_device_bytes: retired_broadcast.stream_peak_device_bytes,
            replica_state_stream_backends: retired_broadcast.stream_backends,
            replica_state_host_fallbacks: retired_broadcast.host_fallbacks,
        });
    }

    let primary_range = ranges[0].clone();
    let replica_ranges = ranges[1..].to_vec();
    let primary_shared = Mutex::new(&mut *primary);
    let reduction_gate = (
        Mutex::new(OrderedGradientReductionGate::default()),
        Condvar::new(),
    );
    let pending_ticket = pending_broadcast.take();
    let ready_replicas = if pending_ticket.is_some() {
        anyhow::ensure!(
            replica_graphs.is_none(),
            "replica scheduler cannot own ready graphs and an in-flight broadcast simultaneously"
        );
        None
    } else {
        let replicas = replica_graphs
            .take()
            .context("replica scheduler has neither ready graphs nor an in-flight broadcast")?;
        anyhow::ensure!(
            replicas.len() == replica_count,
            "replica scheduler owns {} ready graphs; expected {replica_count}",
            replicas.len()
        );
        Some(replicas)
    };
    let window_replica_count = replica_ranges.len();
    let (
        primary_result,
        mut replica_results,
        idle_replicas,
        retired_broadcast,
        replica_broadcast_compute_handoffs,
    ) = std::thread::scope(|scope| -> Result<_> {
        // Start the primary next-window shard immediately. If the preceding
        // optimizer generation is still being drained, its detached source
        // is owned by `pending_broadcast`; forward/backward may safely read
        // those parameters while the ticket runs in parallel.
        let primary_handle = {
            let primary_shared = &primary_shared;
            let reduction_gate = &reduction_gate;
            scope.spawn(move || -> Result<_> {
                let mut primary = primary_shared.lock().map_err(|_| {
                    anyhow::anyhow!("primary Vulkan training graph lock was poisoned")
                })?;
                match primary
                    .train_zero_state_raw_token_labeled_sequences_budgeted_with_update_mode(
                        batch,
                        &inputs[primary_range],
                        objective,
                        hyper,
                        HierarchosTokenTapeUpdateMode::BeginAccumulation,
                        policy,
                        readback,
                    ) {
                    Ok(result) => Ok(result),
                    Err(err) => {
                        let (gate_lock, gate_cv) = reduction_gate;
                        let mut gate = gate_lock.lock().map_err(|_| {
                            anyhow::anyhow!("gradient reduction ordering gate was poisoned")
                        })?;
                        gate.failed = true;
                        gate_cv.notify_all();
                        Err(err)
                    }
                }
            })
        };

        let mut replica_handles = Vec::with_capacity(window_replica_count);
        let mut idle_replicas = Vec::with_capacity(replica_count - window_replica_count);
        let mut broadcast_compute_handoffs = 0usize;
        let cancel_reductions = || {
            let (gate_lock, gate_cv) = &reduction_gate;
            let mut gate = gate_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            gate.failed = true;
            gate_cv.notify_all();
        };
        let spawn_replica = |replica_index: usize, mut replica: HierarchosTrainingGraph| {
            let range = replica_ranges[replica_index].clone();
            let primary_shared = &primary_shared;
            let reduction_gate = &reduction_gate;
            scope.spawn(move || -> Result<_> {
                        let replica_result = match replica
                            .train_zero_state_raw_token_labeled_sequences_budgeted_with_update_mode(
                                batch,
                                &inputs[range],
                                objective,
                                hyper,
                                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                                policy,
                                readback,
                            ) {
                            Ok(result) => result,
                            Err(err) => {
                                let (gate_lock, gate_cv) = reduction_gate;
                                let mut gate = gate_lock.lock().map_err(|_| {
                                    anyhow::anyhow!(
                                        "gradient reduction ordering gate was poisoned"
                                    )
                                })?;
                                gate.failed = true;
                                gate_cv.notify_all();
                                return Err(err);
                            }
                        };

                        // Detach the exact canonical gradient buffers from the
                        // non-Sync replica graph as soon as backward finishes.
                        // Ordered transport/reduction below now depends only on
                        // cloned Vulkan buffer identities plus fixed-size merge
                        // metadata, which is the ownership seam needed for
                        // source-side DMA prefetch and persistent replica lanes.
                        let gradient_source = match replica
                            .full_model_pending_gradient_transport_source()
                        {
                            Ok(source) => source,
                            Err(err) => {
                                let (gate_lock, gate_cv) = reduction_gate;
                                let mut gate = gate_lock.lock().map_err(|_| {
                                    anyhow::anyhow!(
                                        "gradient reduction ordering gate was poisoned"
                                    )
                                })?;
                                gate.failed = true;
                                gate_cv.notify_all();
                                return Err(err);
                            }
                        };

                        let (gate_lock, gate_cv) = reduction_gate;
                        let mut gate = gate_lock.lock().map_err(|_| {
                            anyhow::anyhow!("gradient reduction ordering gate was poisoned")
                        })?;
                        while !gate.failed && gate.next_replica != replica_index {
                            gate = gate_cv.wait(gate).map_err(|_| {
                                anyhow::anyhow!("gradient reduction ordering gate was poisoned")
                            })?;
                        }
                        if gate.failed {
                            bail!(
                                "gradient reduction for replica {replica_index} cancelled after an earlier worker failure"
                            );
                        }
                        drop(gate);

                        let stream_result = (|| -> Result<_> {
                            let mut primary = primary_shared.lock().map_err(|_| {
                                anyhow::anyhow!("primary Vulkan training graph lock was poisoned")
                            })?;
                            primary.accumulate_full_model_pending_gradients_streamed_from_source(
                                &gradient_source,
                                gradient_stream_chunk_values,
                            )
                        })();

                        let mut gate = gate_lock.lock().map_err(|_| {
                            anyhow::anyhow!("gradient reduction ordering gate was poisoned")
                        })?;
                        match stream_result {
                            Ok(stream) => {
                                gate.next_replica = gate
                                    .next_replica
                                    .checked_add(1)
                                    .context("gradient reduction replica-order counter overflow")?;
                                gate_cv.notify_all();
                                Ok((replica_index, replica, replica_result, stream))
                            }
                            Err(err) => {
                                gate.failed = true;
                                gate_cv.notify_all();
                                Err(err)
                            }
                        }
                    })
        };

        let mut retired_broadcast = ReplicaStateBroadcastSummary::default();
        if let Some(ticket) = pending_ticket {
            anyhow::ensure!(
                ticket.replica_count == replica_count,
                "replica broadcast ticket owns {} graphs; scheduler expects {replica_count}",
                ticket.replica_count
            );
            let mut seen = vec![false; replica_count];
            for _ in 0..replica_count {
                let completion = match ticket.recv_completion() {
                    Ok(completion) => completion,
                    Err(err) => {
                        cancel_reductions();
                        return Err(err);
                    }
                };
                if completion.replica_index >= replica_count {
                    cancel_reductions();
                    bail!(
                        "replica-state worker returned invalid index {} for {} replicas",
                        completion.replica_index,
                        replica_count
                    );
                }
                if seen[completion.replica_index] {
                    cancel_reductions();
                    bail!(
                        "replica-state worker returned duplicate graph index {}",
                        completion.replica_index
                    );
                }
                seen[completion.replica_index] = true;
                match completion.result {
                    Ok(worker_summary) => {
                        if let Err(err) = merge_replica_state_broadcast_summary(
                            &mut retired_broadcast,
                            worker_summary,
                        ) {
                            cancel_reductions();
                            return Err(err);
                        }
                    }
                    Err(err) => {
                        cancel_reductions();
                        return Err(err.context(format!(
                            "replica-state broadcast worker {} failed before next-window compute",
                            completion.replica_index
                        )));
                    }
                }
                if completion.replica_index < window_replica_count {
                    replica_handles.push(spawn_replica(completion.replica_index, completion.graph));
                    broadcast_compute_handoffs = match broadcast_compute_handoffs.checked_add(1) {
                        Some(next) => next,
                        None => {
                            cancel_reductions();
                            bail!("replica broadcast-to-compute handoff counter overflow");
                        }
                    };
                } else {
                    idle_replicas.push((completion.replica_index, completion.graph));
                }
            }
        } else if let Some(replicas) = ready_replicas {
            for (replica_index, replica) in replicas.into_iter().enumerate() {
                if replica_index < window_replica_count {
                    replica_handles.push(spawn_replica(replica_index, replica));
                } else {
                    idle_replicas.push((replica_index, replica));
                }
            }
        }
        let primary_result = primary_handle
            .join()
            .map_err(|_| anyhow::anyhow!("primary Vulkan training worker panicked"))??;

        let mut replica_results = Vec::with_capacity(replica_handles.len());
        for handle in replica_handles {
            replica_results.push(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("Vulkan replica training worker panicked"))??,
            );
        }
        Ok((
            primary_result,
            replica_results,
            idle_replicas,
            retired_broadcast,
            broadcast_compute_handoffs,
        ))
    })?;
    drop(primary_shared);

    replica_results.sort_by_key(|entry| entry.0);
    let active_replica_count = replica_results.len();
    anyhow::ensure!(
        active_replica_count == window_replica_count,
        "next-window replica scheduler ran {active_replica_count} shards; expected {window_replica_count}"
    );
    let mut returned_replicas = (0..replica_count).map(|_| None).collect::<Vec<_>>();
    for (replica_index, replica) in idle_replicas {
        anyhow::ensure!(
            returned_replicas[replica_index].is_none(),
            "idle replica graph {replica_index} was returned twice"
        );
        returned_replicas[replica_index] = Some(replica);
    }
    let mut queue_submissions = u64::from(primary_result.queue_submissions);
    let mut losses = primary_result
        .sequences
        .into_iter()
        .flat_map(|sequence| sequence.losses)
        .collect::<Vec<_>>();
    let mut gradient_stream_chunks = 0usize;
    let mut gradient_stream_values = 0usize;
    let mut gradient_stream_pipeline_slots = 0usize;
    let mut gradient_stream_persistent_reuses = 0usize;
    let mut gradient_stream_peak_host_bytes = 0usize;
    let mut gradient_stream_peak_device_bytes = 0usize;
    let mut gradient_stream_peak_host_heap_bytes = 0usize;
    let mut gradient_stream_backends = HashSet::new();
    for (replica_index, replica, result, stream) in replica_results {
        anyhow::ensure!(
            returned_replicas[replica_index].is_none(),
            "active replica graph {replica_index} was returned twice"
        );
        returned_replicas[replica_index] = Some(replica);
        queue_submissions = queue_submissions
            .checked_add(u64::from(result.queue_submissions))
            .context("multi-device queue-submission count overflow")?;
        losses.extend(
            result
                .sequences
                .into_iter()
                .flat_map(|sequence| sequence.losses),
        );
        gradient_stream_backends.insert(stream.backend);
        gradient_stream_chunks = gradient_stream_chunks
            .checked_add(stream.chunk_count)
            .context("multi-device gradient-stream chunk counter overflow")?;
        gradient_stream_values = gradient_stream_values
            .checked_add(stream.value_count)
            .context("multi-device gradient-stream value counter overflow")?;
        gradient_stream_pipeline_slots = gradient_stream_pipeline_slots.max(stream.pipeline_slots);
        if stream.persistent_transport_reused {
            gradient_stream_persistent_reuses = gradient_stream_persistent_reuses
                .checked_add(1)
                .context("multi-device persistent transport reuse counter overflow")?;
        }
        gradient_stream_peak_host_bytes =
            gradient_stream_peak_host_bytes.max(stream.peak_host_gradient_bytes);
        gradient_stream_peak_device_bytes =
            gradient_stream_peak_device_bytes.max(stream.peak_device_gradient_bytes);
        gradient_stream_peak_host_heap_bytes =
            gradient_stream_peak_host_heap_bytes.max(stream.peak_host_heap_gradient_bytes);
        queue_submissions = queue_submissions
            .checked_add(stream.queue_submissions as u64)
            .context("multi-device reduction submission count overflow")?;
    }
    *replica_graphs = Some(
        returned_replicas
            .into_iter()
            .enumerate()
            .map(|(replica_index, graph)| {
                graph.with_context(|| {
                    format!(
                        "replica scheduler lost graph {replica_index} after next-window compute"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?,
    );
    queue_submissions = queue_submissions
        .checked_add(retired_broadcast.queue_submissions)
        .context("multi-device retired broadcast submission count overflow")?;
    let optimizer = primary.finish_full_model_accumulation(hyper)?;
    queue_submissions = queue_submissions
        .checked_add(1)
        .context("multi-device optimizer submission count overflow")?;

    launch_replica_state_broadcast(
        primary,
        broadcast_pool,
        replica_graphs,
        pending_broadcast,
        replica_count,
        active_replica_count,
        gradient_stream_chunk_values,
    )?;

    Ok(BudgetedWindowTrainSummary {
        optimizer_step: optimizer.step,
        queue_submissions,
        losses,
        gradient_reductions: active_replica_count,
        replica_broadcasts: replica_count,
        replica_broadcast_compute_handoffs,
        gradient_stream_chunks,
        gradient_stream_values,
        gradient_stream_pipeline_slots,
        gradient_stream_persistent_reuses,
        gradient_stream_peak_host_bytes,
        gradient_stream_peak_device_bytes,
        gradient_stream_peak_host_heap_bytes,
        gradient_stream_backends,
        replica_state_stream_chunks: retired_broadcast.stream_chunks,
        replica_state_stream_values: retired_broadcast.stream_values,
        replica_state_stream_pipeline_slots: retired_broadcast.stream_pipeline_slots,
        replica_state_stream_persistent_reuses: retired_broadcast.stream_persistent_reuses,
        replica_state_stream_peak_host_bytes: retired_broadcast.stream_peak_host_bytes,
        replica_state_stream_peak_device_bytes: retired_broadcast.stream_peak_device_bytes,
        replica_state_stream_backends: retired_broadcast.stream_backends,
        replica_state_host_fallbacks: retired_broadcast.host_fallbacks,
    })
}

fn main() -> Result<()> {
    let trainer = std::thread::Builder::new()
        .name("hierarchos-vulkan-trainer".to_owned())
        .stack_size(TRAINER_HOST_STACK_BYTES)
        .spawn(run)
        .context("spawning Hierarchos Vulkan trainer host thread")?;
    trainer.join().map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic payload");
        anyhow::anyhow!("Hierarchos Vulkan trainer thread panicked: {message}")
    })?
}

fn run() -> Result<()> {
    let mut args = parse_args()?;
    if let Some(precision) = args.training_precision_override.as_deref() {
        // The training graph and every replica intentionally consume one
        // process-wide precision policy. Apply the CLI override before Vulkan
        // device discovery or graph construction so all replicas inherit the
        // same numerical contract and the existing environment-based lower
        // layers remain reusable by benchmarks and diagnostic binaries.
        std::env::set_var(HIERARCHOS_VULKAN_TRAINING_PRECISION_ENV, precision);
    }
    let requested_gradient_stream_chunk_values = args.gradient_stream_chunk_values;
    anyhow::ensure!(
        args.model_dir != args.output_dir,
        "--output must be distinct from the training source package so checkpoint export is atomic"
    );
    if args.resume_from_checkpoint {
        validate_exact_resume_package(&args.model_dir)?;
    }
    let loaded_dataset = load_training_dataset(&args.dataset_path)?;
    let dataset = loaded_dataset.rows;
    let dataset_lengths = dataset_row_lengths(&dataset)?;
    let full_batch_count = dataset.len() / args.batch_size;
    anyhow::ensure!(
        full_batch_count > 0,
        "dataset has {} rows, fewer than --batch-size {}",
        dataset.len(),
        args.batch_size
    );
    let dropped_rows = dataset.len() - full_batch_count * args.batch_size;
    let gui_total_steps = u64::try_from(full_batch_count)
        .context("native full batch count exceeds u64")?
        .checked_mul(args.epochs)
        .context("native GUI total-step counter overflow")?;

    let device_catalog = VulkanDevice::enumerate_compute_devices()
        .context("enumerating Vulkan transport capabilities")?;
    let config = ModelConfig::from_model_dir(&args.model_dir)
        .with_context(|| format!("loading model config from {}", args.model_dir.display()))?;
    if args.tbptt_chunk_size.is_none() {
        args.tbptt_chunk_size = Some(architecture_default_tbptt_chunk_size(
            &config.architecture_revision,
        ));
    }
    let run_identity = native_run_identity(
        &args,
        &dataset,
        &loaded_dataset.identity,
        &config,
        full_batch_count,
    )
    .context("building content-addressed native run identity")?;
    if args.resume_from_checkpoint {
        validate_native_run_identity_package(&args.model_dir, &run_identity)
            .context("validating exact native resume identity")?;
    }
    let mut opaque_external_probed_pairs = HashSet::new();
    let grouped_device_views = if let Some(indices) = args
        .device_indices
        .as_ref()
        .filter(|indices| indices.len() > 1)
    {
        let primary_info = device_catalog
            .iter()
            .find(|device| device.index == indices[0]);
        let shared_group_candidate = primary_info.is_some_and(|primary| {
            indices[1..].iter().all(|replica_index| {
                device_catalog
                    .iter()
                    .find(|device| device.index == *replica_index)
                    .is_some_and(|replica| primary.device_group_transport_candidate_with(replica))
            })
        });
        if !shared_group_candidate {
            if let Some(primary) = primary_info {
                for &replica_index in &indices[1..] {
                    let Some(replica) = device_catalog
                        .iter()
                        .find(|device| device.index == replica_index)
                    else {
                        continue;
                    };
                    if !primary.opaque_external_transport_candidate_with(replica) {
                        continue;
                    }
                    match VulkanDevice::probe_opaque_external_transport_indices(
                        primary.index,
                        replica.index,
                    ) {
                        Ok(probe) => {
                            eprintln!(
                                "vulkan_gradient_transport_probe candidate=opaque-external-memory source_device={} destination_device={} handle={} synchronized_roundtrip={} payload_bytes={}",
                                primary.index,
                                replica.index,
                                probe.handle_name,
                                probe.synchronized_roundtrip,
                                probe.payload_bytes,
                            );
                            opaque_external_probed_pairs.insert(replica.index);
                        }
                        Err(err) => {
                            eprintln!(
                                "vulkan_gradient_transport_probe candidate=opaque-external-memory source_device={} destination_device={} live_probe=failed fallback=host-visible-staged-v2-pipelined reason={err:#}",
                                primary.index,
                                replica.index,
                            );
                        }
                    }
                }
            }
        }
        if shared_group_candidate {
            match VulkanDevice::new_device_group_with_indices(indices) {
                Ok(views) => {
                    eprintln!(
                        "vulkan_gradient_transport_probe candidate=device-group logical_device=created physical_device_count={} device_mask=0x{:08x} queue_lanes={} timeline_semaphore={}",
                        views[0].device_group_physical_device_count(),
                        views[0].device_group_mask(),
                        views[0].queue_lane_count(),
                        views[0].device_group_timeline_semaphore_enabled(),
                    );
                    Some(views)
                }
                Err(err) => {
                    eprintln!(
                        "vulkan_gradient_transport_probe candidate=device-group logical_device=failed fallback=host-visible-staged-v2-pipelined reason={err:#}"
                    );
                    None
                }
            }
        } else {
            eprintln!(
                "vulkan_gradient_transport_probe candidate=device-group topology=unavailable fallback=host-visible-staged-v2-pipelined"
            );
            None
        }
    } else {
        None
    };
    let device = if let Some(views) = grouped_device_views.as_ref() {
        views[0].clone()
    } else if let Some(indices) = args.device_indices.as_ref() {
        VulkanDevice::new_with_index(indices[0]).with_context(|| {
            format!(
                "initializing primary Vulkan training device index {}",
                indices[0]
            )
        })?
    } else {
        match args.device_index {
            Some(index) => VulkanDevice::new_with_index(index)
                .with_context(|| format!("initializing Vulkan training device index {index}"))?,
            None => VulkanDevice::new().context("initializing Vulkan training device")?,
        }
    };
    let anticipated_transport_backends = if let Some(indices) = args
        .device_indices
        .as_ref()
        .filter(|indices| indices.len() > 1)
    {
        if grouped_device_views.is_some() {
            vec![VulkanGradientTransportBackend::DeviceGroup; indices.len() - 1]
        } else {
            let primary = device_catalog
                .iter()
                .find(|candidate| candidate.index == indices[0])
                .context("primary Vulkan device disappeared before memory preflight")?;
            indices[1..]
                .iter()
                .map(|replica_index| -> Result<VulkanGradientTransportBackend> {
                    let replica = device_catalog
                        .iter()
                        .find(|candidate| candidate.index == *replica_index)
                        .with_context(|| {
                            format!(
                                "Vulkan replica device {replica_index} disappeared before memory preflight"
                            )
                        })?;
                    // Reserve opaque-external steady-state residency whenever
                    // the pair can legally instantiate that path, even if the
                    // early smoke probe failed. Runtime may retry allocation on
                    // first use; budgeting the candidate is safer than letting
                    // that upgrade appear as unplanned primary-local memory.
                    Ok(if primary.opaque_external_transport_candidate_with(replica) {
                        VulkanGradientTransportBackend::OpaqueExternalMemory
                    } else {
                        VulkanGradientTransportBackend::HostVisibleStagedV2Pipelined
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
    } else {
        Vec::new()
    };
    let anticipated_transport_lanes = anticipated_transport_backends.len();
    let joint_runtime_tokens_per_sequence = dataset
        .iter()
        .map(|row| row.input_ids.len())
        .max()
        .context("joint runtime scheduler could not determine dataset token width")?;
    let mut joint_memory_preflight = None;
    let mut phase_aware_tape_geometry = None;
    let mut joint_runtime_autotuner = None;
    let mut joint_runtime_profile_identity = None;
    let mut loaded_joint_runtime_profile = None;
    let mut loaded_joint_runtime_profile_source = None;
    let mut persistent_profile_transport_ceiling = None;
    let mut joint_runtime_profile_scored_windows_since_persist = 0u64;
    let mut last_persisted_joint_runtime_winner = None;
    if anticipated_transport_lanes > 0 {
        let selected_indices = args
            .device_indices
            .as_ref()
            .context("multi-device memory preflight is missing selected device indices")?;
        let profile_key = joint_runtime_profile_key(
            &config,
            &args,
            joint_runtime_tokens_per_sequence,
            &device_catalog,
            selected_indices,
            &anticipated_transport_backends,
        )?;
        loaded_joint_runtime_profile = if std::env::var_os(
            HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_DISABLE_ENV,
        )
        .is_some()
        {
            if args.joint_runtime_profile.is_some() {
                bail!(
                    "--joint-runtime-profile cannot be combined with {HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_DISABLE_ENV}"
                );
            }
            None
        } else if let Some(explicit_profile_path) = args.joint_runtime_profile.as_ref() {
            let profile = load_joint_runtime_profile_path(explicit_profile_path, &profile_key)?
                .with_context(|| {
                    format!(
                        "explicit Vulkan joint-runtime profile {} does not match the current architecture/batch/token/device/driver/transport identity",
                        explicit_profile_path.display()
                    )
                })?;
            loaded_joint_runtime_profile_source = Some(explicit_profile_path.clone());
            Some(profile)
        } else {
            let mut loaded = None;
            let mut profile_dirs = vec![args.output_dir.as_path()];
            if args.output_dir != args.model_dir {
                profile_dirs.push(args.model_dir.as_path());
            }
            for profile_dir in profile_dirs {
                match load_joint_runtime_profile(profile_dir, &profile_key) {
                    Ok(Some(profile)) => {
                        loaded_joint_runtime_profile_source =
                            Some(profile_dir.join(JOINT_RUNTIME_PROFILE_FILENAME));
                        loaded = Some(profile);
                        break;
                    }
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!(
                            "vulkan_joint_runtime_profile load=ignored source={} reason={err:#}",
                            profile_dir.join(JOINT_RUNTIME_PROFILE_FILENAME).display()
                        );
                    }
                }
            }
            loaded
        };
        if let Some(profile) = loaded_joint_runtime_profile.as_ref() {
            let ceiling = requested_gradient_stream_chunk_values
                .min(profile.winning_arm.gradient_stream_chunk_values)
                .max(1);
            persistent_profile_transport_ceiling = Some(ceiling);
            last_persisted_joint_runtime_winner = Some(profile.winning_arm);
            eprintln!(
                "vulkan_joint_runtime_profile load=matched source={} learned_transport_chunk={} requested_transport_chunk={} planner_transport_ceiling={}",
                loaded_joint_runtime_profile_source
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "unknown".to_owned()),
                profile.winning_arm.gradient_stream_chunk_values,
                requested_gradient_stream_chunk_values,
                ceiling,
            );
        }
        joint_runtime_profile_identity = Some(profile_key);
        let mut adapter_constraints = Vec::with_capacity(selected_indices.len());
        let primary_budget = device
            .memory_budget()
            .context("querying primary Vulkan memory budget before graph allocation")?;
        adapter_constraints.push(
            HierarchosJointTrainingAdapterConstraint::from_preconstruction_budget(
                device.physical_device_index(),
                primary_budget,
            ),
        );
        for (replica_position, &replica_index) in selected_indices[1..].iter().enumerate() {
            let replica_budget = if let Some(views) = grouped_device_views.as_ref() {
                views[replica_position + 1].memory_budget().with_context(|| {
                    format!(
                        "querying Vulkan device-group replica memory budget for physical device {replica_index}"
                    )
                })?
            } else {
                let replica_budget_device = VulkanDevice::new_with_index(replica_index)
                    .with_context(|| {
                        format!(
                            "initializing Vulkan replica device {replica_index} for pre-allocation memory budgeting"
                        )
                    })?;
                replica_budget_device.memory_budget().with_context(|| {
                    format!(
                        "querying Vulkan replica memory budget for physical device {replica_index}"
                    )
                })?
            };
            adapter_constraints.push(
                HierarchosJointTrainingAdapterConstraint::from_preconstruction_budget(
                    replica_index,
                    replica_budget,
                ),
            );
        }
        let planner_transport_ceiling =
            persistent_profile_transport_ceiling.unwrap_or(requested_gradient_stream_chunk_values);
        let preflight_chunk_values = planner_transport_ceiling
            .min(JOINT_MEMORY_PREFLIGHT_CHUNK_VALUES)
            .max(1);
        let preflight_graph =
            HierarchosTrainingGraph::from_model_package_with_token_frontend_and_optimizer_staging(
                device.clone(),
                &args.model_dir,
                args.batch_size,
                config.max_h_steps,
                config.max_l_steps,
                args.batch_size,
                args.batch_size,
                preflight_chunk_values,
            )
            .context("building minimal-staging graph for phase-aware memory preflight")?;
        let preflight_sequences = args
            .gradient_accumulation_steps
            .min(full_batch_count)
            .max(1);
        let preflight_tokens = joint_runtime_tokens_per_sequence;
        match preflight_graph.plan_joint_training_memory_with_adapter_constraints(
            args.batch_size,
            preflight_sequences,
            preflight_tokens,
            planner_transport_ceiling,
            &anticipated_transport_backends,
            &adapter_constraints,
            HierarchosTapeMemoryPolicy::default(),
        ) {
            Ok(plan) => {
                args.gradient_stream_chunk_values = plan.transport_chunk_values;
                let selected_tape_geometry = PhaseAwareTapeGeometry {
                    sequence_microbatch_size: plan.sequence_microbatch_size,
                    state_checkpoint_stride: plan.state_checkpoint_stride,
                };
                phase_aware_tape_geometry = Some(selected_tape_geometry);
                joint_runtime_autotuner = JointRuntimeAutotuner::new(
                    plan.transport_chunk_values,
                    selected_tape_geometry,
                    preflight_tokens,
                    loaded_joint_runtime_profile.as_ref(),
                    args.lock_joint_runtime_profile,
                );
                if args.lock_joint_runtime_profile
                    && joint_runtime_autotuner
                        .as_ref()
                        .and_then(JointRuntimeAutotuner::locked_arm)
                        .is_none()
                {
                    bail!(
                        "explicit Vulkan joint-runtime profile winner is not safe under the current phase-memory preflight constraints"
                    );
                }
                eprintln!(
                    "vulkan_phase_memory_preflight requested_transport_chunk={} selected_transport_chunk={} sequences={} tokens={} tape_microbatch={} checkpoint_stride={} projected_working_set_bytes={} persistent_transport_bytes={} persistent_broadcast_transport_bytes={} persistent_gradient_cache_bytes={} transient_transport_phase_bytes={} transient_transport_active_lanes={} device_group_lanes={} opaque_external_lanes={} host_staged_lanes={} tape_phase_peak_bytes={} gradient_reduction_phase_peak_bytes={} optimizer_wavefront_broadcast_phase_peak_bytes={} phase_aware_peak_bytes={} scalar_sum_peak_bytes={} mutually_exclusive_bytes_recovered={} working_set_limit_bytes={} constrained_adapters={} limiting_physical_device_index={}",
                    requested_gradient_stream_chunk_values,
                    plan.transport_chunk_values,
                    preflight_sequences,
                    preflight_tokens,
                    plan.sequence_microbatch_size,
                    plan.state_checkpoint_stride,
                    plan.projected_working_set_bytes,
                    plan.persistent_transport_bytes,
                    plan.persistent_broadcast_transport_bytes,
                    plan.persistent_gradient_cache_bytes,
                    plan.transient_transport_phase_bytes,
                    plan.transient_transport_active_lanes,
                    plan.device_group_transport_lanes,
                    plan.opaque_external_transport_lanes,
                    plan.host_staged_transport_lanes,
                    plan.tape_phase_peak_bytes,
                    plan.gradient_reduction_phase_peak_bytes,
                    plan.optimizer_wavefront_broadcast_phase_peak_bytes,
                    plan.phase_aware_peak_bytes,
                    plan.scalar_sum_peak_bytes,
                    plan.mutually_exclusive_bytes_recovered,
                    plan.working_set_limit_bytes,
                    plan.adapter_memory_plans.len(),
                    plan.limiting_physical_device_index,
                );
                let adapter_memory_plan_report = plan
                    .adapter_memory_plans
                    .iter()
                    .map(|adapter| {
                        serde_json::json!({
                            "physical_device_index": adapter.physical_device_index,
                            "driver_budget_bytes": adapter.budget.device_local_budget_bytes,
                            "driver_usage_bytes": adapter.budget.device_local_usage_bytes,
                            "budget_extension_supported": adapter.budget.budget_extension_supported,
                            "working_set_limit_bytes": adapter.working_set_limit_bytes,
                            "resident_without_training_graph_bytes": adapter.resident_without_training_graph_bytes,
                            "projected_working_set_bytes": adapter.projected_working_set_bytes,
                            "persistent_transport_bytes": adapter.persistent_transport_bytes,
                            "transient_transport_phase_bytes": adapter.transient_transport_phase_bytes,
                            "tape_phase_peak_bytes": adapter.tape_phase_peak_bytes,
                            "gradient_reduction_phase_peak_bytes": adapter.gradient_reduction_phase_peak_bytes,
                            "optimizer_broadcast_phase_peak_bytes": adapter.optimizer_broadcast_phase_peak_bytes,
                            "phase_aware_peak_bytes": adapter.phase_aware_peak_bytes,
                            "remaining_headroom_bytes": adapter.remaining_headroom_bytes,
                        })
                    })
                    .collect::<Vec<_>>();
                joint_memory_preflight = Some(serde_json::json!({
                    "scheduler": "phase-concurrency-v4-physical-adapter-vector",
                    "requested_transport_chunk_values": requested_gradient_stream_chunk_values,
                    "persistent_profile_transport_ceiling": persistent_profile_transport_ceiling,
                    "planner_transport_ceiling": planner_transport_ceiling,
                    "persistent_profile_applied": loaded_joint_runtime_profile.is_some(),
                    "selected_transport_chunk_values": plan.transport_chunk_values,
                    "anticipated_transport_lanes": plan.anticipated_transport_lanes,
                    "transport_backends": anticipated_transport_backends
                        .iter()
                        .map(|backend| backend.label())
                        .collect::<Vec<_>>(),
                    "sequence_microbatch_size": plan.sequence_microbatch_size,
                    "sequence_microbatch_count": plan.sequence_microbatch_count,
                    "state_checkpoint_stride": plan.state_checkpoint_stride,
                    "projected_working_set_bytes": plan.projected_working_set_bytes,
                    "resident_without_probe_working_set_bytes": plan.resident_without_probe_working_set_bytes,
                    "persistent_transport_bytes": plan.persistent_transport_bytes,
                    "persistent_broadcast_transport_bytes": plan.persistent_broadcast_transport_bytes,
                    "persistent_gradient_cache_bytes": plan.persistent_gradient_cache_bytes,
                    "transient_transport_phase_bytes": plan.transient_transport_phase_bytes,
                    "transient_transport_active_lanes": plan.transient_transport_active_lanes,
                    "device_group_transport_lanes": plan.device_group_transport_lanes,
                    "opaque_external_transport_lanes": plan.opaque_external_transport_lanes,
                    "host_staged_transport_lanes": plan.host_staged_transport_lanes,
                    "tape_peak_bytes": plan.tape_peak_bytes,
                    "tape_phase_peak_bytes": plan.tape_phase_peak_bytes,
                    "gradient_reduction_phase_peak_bytes": plan.gradient_reduction_phase_peak_bytes,
                    "optimizer_wavefront_broadcast_phase_peak_bytes": plan.optimizer_wavefront_broadcast_phase_peak_bytes,
                    "optimizer_broadcast_phase_peak_bytes": plan.optimizer_broadcast_phase_peak_bytes,
                    "phase_aware_peak_bytes": plan.phase_aware_peak_bytes,
                    "scalar_sum_peak_bytes": plan.scalar_sum_peak_bytes,
                    "mutually_exclusive_bytes_recovered": plan.mutually_exclusive_bytes_recovered,
                    "compute_graph_live_bytes": plan.compute_graph_live_bytes,
                    "gradient_reduction_graph_live_bytes": plan.gradient_reduction_graph_live_bytes,
                    "optimizer_broadcast_graph_live_bytes": plan.optimizer_broadcast_graph_live_bytes,
                    "working_set_limit_bytes": plan.working_set_limit_bytes,
                    "limiting_physical_device_index": plan.limiting_physical_device_index,
                    "adapter_memory_plans": adapter_memory_plan_report,
                }));
            }
            Err(err) => {
                eprintln!(
                    "vulkan_phase_memory_preflight fallback=requested-transport-width reason={err:#}"
                );
            }
        }
        drop(preflight_graph);
    }
    let primary_device_index = device.physical_device_index();
    eprintln!(
        "vulkan_device_role=primary vulkan_device_index={} vulkan_device={}",
        primary_device_index,
        device.name()
    );
    emit_gui_event(
        args.json_events,
        serde_json::json!({
            "event": "training_started",
            "backend": "vulkan",
            "device_index": primary_device_index,
            "device": device.name(),
            "epochs": args.epochs,
            "batches_per_epoch": full_batch_count,
            "total_steps": gui_total_steps,
            "output_dir": args.output_dir.display().to_string(),
            "checkpoint_abi": "pytorch-safetensors-fp32-master",
        }),
    );
    let primary_replica_transport_device = device.clone();
    let mut graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend_and_optimizer_staging(
            device,
            &args.model_dir,
            args.batch_size,
            config.max_h_steps,
            config.max_l_steps,
            args.batch_size,
            args.batch_size,
            args.gradient_stream_chunk_values,
        )
        .context("building full Hierarchos Vulkan training graph")?;
    let (active_trainable_tensors, frozen_trainable_tensors) = graph
        .set_trainable_parameter_prefixes(&args.trainable_prefixes)
        .context("configuring Vulkan trainable-parameter selection")?;
    if !args.trainable_prefixes.is_empty() {
        eprintln!(
            "vulkan_trainable_selection active_tensors={} frozen_tensors={} prefixes={:?}",
            active_trainable_tensors, frozen_trainable_tensors, args.trainable_prefixes
        );
    }
    let training_precision = graph.training_precision_policy();
    let dynamic_loss_scaling_enabled = precision_uses_dynamic_loss_scaling(training_precision);
    let resumed_optimizer = args.resume_from_checkpoint;
    let resume_manifest = if resumed_optimizer {
        Some(
            graph
                .load_training_checkpoint_package_state(&args.model_dir)
                .context("restoring Vulkan optimizer/checkpoint state")?,
        )
    } else {
        None
    };
    if args
        .device_indices
        .as_ref()
        .is_some_and(|indices| indices.len() > 1)
    {
        anyhow::ensure!(
            !resume_manifest
                .as_ref()
                .is_some_and(|manifest| manifest.accumulation_open),
            "multi-device Vulkan resume currently requires a closed optimizer-step boundary"
        );
    }

    let mut replica_graphs = Vec::new();
    let mut replica_transport_devices = Vec::new();
    if let Some(indices) = args
        .device_indices
        .as_ref()
        .filter(|indices| indices.len() > 1)
    {
        let primary_state = graph
            .full_model_replica_state()
            .context("capturing primary state for Vulkan replica initialization")?;
        for (replica_position, &index) in indices[1..].iter().enumerate() {
            let replica_device = if let Some(views) = grouped_device_views.as_ref() {
                views[replica_position + 1].clone()
            } else {
                VulkanDevice::new_with_index(index)
                    .with_context(|| format!("initializing Vulkan replica device index {index}"))?
            };
            eprintln!(
                "vulkan_device_role=replica vulkan_device_index={} vulkan_device={}",
                replica_device.physical_device_index(),
                replica_device.name()
            );
            replica_transport_devices.push(replica_device.clone());
            let mut replica = HierarchosTrainingGraph::from_model_package_with_token_frontend_and_optimizer_staging(
                    replica_device,
                    &args.model_dir,
                    args.batch_size,
                    config.max_h_steps,
                    config.max_l_steps,
                    args.batch_size,
                    args.batch_size,
                    args.gradient_stream_chunk_values,
                )
                .with_context(|| format!("building Vulkan training replica on device {index}"))?;
            replica
                .load_full_model_replica_state(&primary_state)
                .with_context(|| {
                    format!("synchronizing Vulkan training replica on device {index}")
                })?;
            replica
                .set_trainable_parameter_prefixes(&args.trainable_prefixes)
                .with_context(|| {
                    format!("configuring trainable selection on Vulkan replica device {index}")
                })?;
            replica_graphs.push(replica);
        }
    }
    let replica_count = replica_graphs.len();
    let replica_worker_pool = ReplicaWorkerPool::new(
        replica_graphs,
        &primary_replica_transport_device,
        &replica_transport_devices,
        args.gradient_stream_chunk_values,
    )?;
    let predeclared_retirement_timeline_lanes =
        replica_worker_pool.predeclared_retirement_timeline_lanes();
    let mut pending_replica_broadcast: Option<ReplicaStateBroadcastTicket> = None;

    let objective = HierarchosLabeledSequenceObjective {
        z_loss_weight: args.z_loss_weight,
        ponder_loss_weight: args.ponder_loss_weight,
        commitment_loss_weight: args.commitment_loss_weight,
        max_ce_loss_for_backward: args.max_ce_loss_for_backward,
        max_ponder_cost_for_backward: args.max_ponder_cost_for_backward,
        max_commitment_cost_for_backward: args.max_commitment_cost_for_backward,
    };
    let updates_per_epoch =
        update_steps_per_epoch(full_batch_count, args.gradient_accumulation_steps)?;
    let total_update_steps = args
        .epochs
        .checked_mul(updates_per_epoch)
        .context("total optimizer update-count overflow")?
        .max(1);
    let mut optimizer_step = graph.full_model_optimizer_state()?.step;
    let resumed_session = resume_manifest
        .as_ref()
        .and_then(|manifest| manifest.training_session.clone());
    let mut skipped_train_batches = resumed_session
        .as_ref()
        .map_or(0, |session| session.skipped_train_batches);

    let source_execution_policy = resumed_session
        .as_ref()
        .and_then(|session| session.execution_policy.as_ref());
    if !dynamic_loss_scaling_enabled
        && source_execution_policy.is_some_and(|policy| policy.loss_scaling.mode == "dynamic")
    {
        bail!(
            "checkpoint carries an active FP16 dynamic GradScaler execution policy, but the current Vulkan graph is {:?}; select HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-parity (or fp16-lm-backward) for an exact AMP continuation",
            training_precision
        );
    }
    let mut loss_scaling = if dynamic_loss_scaling_enabled {
        if let Some(policy) = source_execution_policy {
            anyhow::ensure!(
                policy.loss_scaling.mode == "dynamic",
                "current Vulkan FP16 training requires dynamic loss scaling, but the resumed checkpoint declares loss_scaling mode {:?}",
                policy.loss_scaling.mode
            );
            anyhow::ensure!(
                policy.compute_dtype == "float16" && policy.autocast_enabled,
                "current Vulkan FP16 training requires a float16 autocast execution policy; resumed checkpoint declares dtype={:?} autocast_enabled={}",
                policy.compute_dtype,
                policy.autocast_enabled
            );
            Some(policy.loss_scaling.clone())
        } else {
            Some(fresh_pytorch_grad_scaler_state(args.initial_loss_scale))
        }
    } else {
        None
    };
    let mut dynamic_loss_scaling_device_resident = false;

    if let Some(session) = resumed_session.as_ref() {
        anyhow::ensure!(
            session.optimizer_grouping_version == 2,
            "native trainer requires optimizer grouping v2, checkpoint carries v{}",
            session.optimizer_grouping_version
        );
        if let Some(saved) = session
            .effective_training_config
            .get("gradient_accumulation_steps")
            .and_then(|value| value.as_u64())
        {
            anyhow::ensure!(
                saved == args.gradient_accumulation_steps as u64,
                "resume gradient accumulation {} does not match --gradient-accumulation-steps {}",
                saved,
                args.gradient_accumulation_steps
            );
        }
        if let Some(saved) = session
            .effective_training_config
            .get("persist_state")
            .and_then(|value| value.as_bool())
        {
            anyhow::ensure!(
                saved == args.persist_state,
                "resume persist_state={} does not match current --persist-state policy {}",
                saved,
                args.persist_state
            );
        }
        validate_resume_numerical_policy(&session.effective_training_config, &args)?;
    }

    let mut cursor = if let Some(session) = resumed_session.as_ref() {
        let mut cursor = session
            .data_stream_cursor
            .clone()
            .unwrap_or(new_data_cursor(&args, dataset.len())?);
        if session.data_stream_cursor.is_none() {
            cursor.epoch = session.completed_epoch;
            cursor.batch_cursor = session.mid_epoch_step;
        }
        anyhow::ensure!(
            cursor.epoch == session.completed_epoch,
            "resume data cursor epoch {} disagrees with session completed_epoch {}",
            cursor.epoch,
            session.completed_epoch
        );
        validate_resume_cursor(&cursor, &args, dataset.len())?;
        cursor
    } else {
        let mut cursor = new_data_cursor(&args, dataset.len())?;
        if let Some(manifest) = resume_manifest.as_ref() {
            cursor.epoch = manifest.completed_epoch.unwrap_or(0);
            cursor.batch_cursor = manifest.mid_epoch_step.unwrap_or(0);
        }
        cursor
    };
    anyhow::ensure!(
        cursor.batch_count()? == full_batch_count as u64,
        "native data cursor batch count diverged from the drop-last trainer batch count"
    );
    anyhow::ensure!(
        cursor.batch_cursor < full_batch_count as u64 || cursor.batch_cursor == 0,
        "resume batch cursor {} is not a runnable drop-last batch",
        cursor.batch_cursor
    );
    let resumed_running_carriers = if args.persist_state && cursor.batch_cursor > 0 {
        let manifest = resume_manifest
            .as_ref()
            .context("mid-epoch persisted-state resume is missing a training manifest")?;
        let replay_file = manifest
            .portable_replay_file
            .as_deref()
            .context("mid-epoch persisted-state resume is missing portable_replay_file")?;
        anyhow::ensure!(
            Path::new(replay_file).components().count() == 1,
            "portable replay file must be a package-local filename"
        );
        let replay_tensor_path = manifest
            .portable_replay_tensor_file
            .as_deref()
            .map(|member| {
                anyhow::ensure!(
                    Path::new(member).components().count() == 1,
                    "portable replay tensor file must be a package-local filename"
                );
                Ok(args.model_dir.join(member))
            })
            .transpose()?;
        let carriers = read_portable_running_carriers(
            &args.model_dir.join(replay_file),
            replay_tensor_path.as_deref(),
        )?
        .context("mid-epoch persisted-state resume replay does not contain running_states")?;
        graph
            .restore_portable_running_carriers(args.batch_size, &carriers)
            .context("restoring portable H/L/context/ROSA running carriers")?;
        Some(carriers)
    } else {
        None
    };
    let portable_ltm_state = if args.persist_state {
        Some(if let Some(carriers) = resumed_running_carriers.as_ref() {
            carriers.ltm_state.clone()
        } else {
            read_model_ltm_running_state(&args.model_dir.join("model.safetensors"))
                .context("loading portable LTM metadata for Vulkan running-state checkpoints")?
        })
    } else {
        None
    };
    if let Some(manifest) = resume_manifest.as_ref() {
        let accumulation_offset = cursor.batch_cursor as usize % args.gradient_accumulation_steps;
        anyhow::ensure!(
            manifest.accumulation_open == (accumulation_offset != 0),
            "resume gradient-window state does not match the native data cursor/accumulation geometry"
        );
    }
    let amp_rehydrate_submissions = if resume_manifest
        .as_ref()
        .is_some_and(|manifest| manifest.accumulation_open)
    {
        if let Some(loss_scaling) = loss_scaling.as_mut() {
            graph
                .rehydrate_full_model_accumulation_for_dynamic_loss_scaling(loss_scaling)
                .context(
                    "rehydrating portable pending gradients into the resumed GradScaler domain",
                )?
        } else {
            0
        }
    } else {
        0
    };

    let mut schedule = resumed_session
        .as_ref()
        .and_then(|session| session.main_lr_scheduler.clone())
        .unwrap_or_else(|| new_lr_schedule(&args, total_update_steps, u64::from(optimizer_step)));
    if schedule.enabled {
        let schedule_step = schedule
            .step
            .context("enabled resumed LR scheduler is missing step")?;
        anyhow::ensure!(
            schedule_step == u64::from(optimizer_step),
            "resume scheduler step {} disagrees with optimizer step {}",
            schedule_step,
            optimizer_step
        );
        let _ = schedule_live_lr(&schedule, args.learning_rate)?;
    }

    let context_len = args
        .batch_size
        .checked_mul(config.context_dim)
        .context("initial context shape overflow")?;
    let mut previous_context = resumed_running_carriers.as_ref().map_or_else(
        || vec![0.0f32; context_len],
        |state| state.previous_context.values.clone(),
    );
    let mut target_context = resumed_running_carriers.as_ref().map_or_else(
        || vec![0.0f32; context_len],
        |state| state.target_context.values.clone(),
    );
    let mut carried_h_state = resumed_running_carriers
        .as_ref()
        .map(|state| state.h_state.values.clone());
    let mut carried_l_state = resumed_running_carriers
        .as_ref()
        .map(|state| state.l_state.values.clone());
    let mut total_queue_submissions = u64::from(amp_rehydrate_submissions);
    let mut total_optimizer_wavefront_windows = 0u64;
    let mut total_optimizer_wavefront_ranges = 0u64;
    let mut total_gradient_reductions = 0u64;
    let mut total_replica_broadcasts = 0u64;
    let mut total_replica_broadcast_compute_handoffs = 0u64;
    let mut total_gradient_stream_chunks = 0u64;
    let mut total_gradient_stream_values = 0u64;
    let mut max_gradient_stream_pipeline_slots = 0usize;
    let mut total_gradient_stream_persistent_reuses = 0u64;
    let mut peak_gradient_stream_host_bytes = 0usize;
    let mut peak_gradient_stream_device_bytes = 0usize;
    let mut peak_gradient_stream_host_heap_bytes = 0usize;
    let mut observed_gradient_stream_backends = HashSet::new();
    let mut total_replica_state_stream_chunks = 0u64;
    let mut total_replica_state_stream_values = 0u64;
    let mut max_replica_state_stream_pipeline_slots = 0usize;
    let mut total_replica_state_stream_persistent_reuses = 0u64;
    let mut peak_replica_state_stream_host_bytes = 0usize;
    let mut peak_replica_state_stream_device_bytes = 0usize;
    let mut observed_replica_state_stream_backends = HashSet::new();
    let mut total_replica_state_host_fallbacks = 0u64;
    let mut epoch_loss_sum = 0.0f64;
    let mut epoch_loss_values = 0usize;
    let start_epoch = cursor.epoch;

    while cursor.epoch < args.epochs {
        let epoch = cursor.epoch;
        let batch_index = usize::try_from(cursor.batch_cursor)
            .context("native batch cursor exceeds host usize range")?;
        if !args.persist_state && batch_index % args.gradient_accumulation_steps == 0 {
            let window_batch_count = args
                .gradient_accumulation_steps
                .min(full_batch_count - batch_index);
            let mut window_cursor = cursor.clone();
            let mut packed_batches = Vec::with_capacity(window_batch_count);
            for window_batch_index in 0..window_batch_count {
                let batch_indices = window_cursor
                    .current_batch_indices(Some(&dataset_lengths))?
                    .context("native data cursor has no runnable budgeted batch")?;
                anyhow::ensure!(
                    batch_indices.len() == args.batch_size,
                    "drop-last native trainer produced a partial budgeted batch"
                );
                let batch_rows = batch_indices
                    .iter()
                    .map(|&index| &dataset[index])
                    .collect::<Vec<_>>();
                packed_batches.push(pack_batch(&batch_rows, config.vocab_size)?);
                let rolled_epoch = window_cursor.advance_batch()?;
                anyhow::ensure!(
                    !rolled_epoch || window_batch_index + 1 == window_batch_count,
                    "budgeted optimizer window crossed an epoch boundary before its final batch"
                );
            }

            let zero_context = vec![0.0f32; context_len];
            let labeled_inputs = packed_batches
                .iter()
                .map(|packed| HierarchosRawTokenLabeledSequenceInput {
                    tokens: packed.tokens,
                    input_ids: &packed.input_ids,
                    labels: &packed.labels,
                    attention_mask: Some(&packed.attention_mask),
                    loss_weights: Some(&packed.loss_weights),
                    initial_previous_context: &zero_context,
                    initial_target_context: &zero_context,
                    global_pos_offset: 0,
                    reset_rosa_at_start: true,
                    pytorch_tbptt_chunk_size: args.tbptt_chunk_size,
                })
                .collect::<Vec<_>>();
            let live_lr = schedule_live_lr(&schedule, args.learning_rate)?;
            let hyper = AdamWHyperParams {
                lr: live_lr,
                beta1: args.beta1,
                beta2: args.beta2,
                eps: args.eps,
                weight_decay: args.weight_decay,
            };
            let runtime_selection = joint_runtime_autotuner.as_mut().map(|autotuner| {
                let (arm_index, arm, score_window) = autotuner.select();
                let lane_workload_weights =
                    autotuner.lane_workload_weights(arm_index, replica_count + 1);
                (arm_index, arm, lane_workload_weights, score_window)
            });
            let (
                runtime_arm_index,
                active_gradient_stream_chunk_values,
                active_tape_geometry,
                optimizer_broadcast_overlap,
                active_lane_workload_weights,
                score_runtime_window,
            ) = if let Some((arm_index, arm, lane_workload_weights, score_window)) =
                runtime_selection
            {
                (
                    Some(arm_index),
                    arm.gradient_stream_chunk_values,
                    Some(arm.tape_geometry),
                    arm.optimizer_broadcast_overlap,
                    Some(lane_workload_weights),
                    score_window,
                )
            } else {
                (
                    None,
                    args.gradient_stream_chunk_values,
                    phase_aware_tape_geometry,
                    true,
                    None,
                    false,
                )
            };
            let runtime_latency_before =
                if score_runtime_window {
                    Some(graph.submission_arena_stats().context(
                        "snapshotting Vulkan submission latency before joint runtime arm",
                    )?)
                } else {
                    None
                };
            if score_runtime_window {
                primary_replica_transport_device
                    .set_scheduler_kernel_timestamp_collection_enabled(true);
                for replica_device in &replica_transport_devices {
                    replica_device.set_scheduler_kernel_timestamp_collection_enabled(true);
                }
            }
            let runtime_window_started = Instant::now();
            let result = if let Some(loss_scaling) = loss_scaling.as_mut() {
                let result = train_budgeted_window(
                    &mut graph,
                    &replica_worker_pool,
                    &mut pending_replica_broadcast,
                    replica_count,
                    args.batch_size,
                    &labeled_inputs,
                    objective,
                    hyper,
                    args.grad_clip,
                    active_gradient_stream_chunk_values,
                    active_tape_geometry,
                    optimizer_broadcast_overlap,
                    active_lane_workload_weights.as_deref(),
                    score_runtime_window,
                    Some(DynamicBudgetedWindow {
                        loss_scaling,
                        device_resident_backward_scale: dynamic_loss_scaling_device_resident,
                        optimizer_step_before: optimizer_step,
                    }),
                );
                if result.is_ok() {
                    dynamic_loss_scaling_device_resident = true;
                }
                result
            } else {
                train_budgeted_window(
                    &mut graph,
                    &replica_worker_pool,
                    &mut pending_replica_broadcast,
                    replica_count,
                    args.batch_size,
                    &labeled_inputs,
                    objective,
                    hyper,
                    args.grad_clip,
                    active_gradient_stream_chunk_values,
                    active_tape_geometry,
                    optimizer_broadcast_overlap,
                    active_lane_workload_weights.as_deref(),
                    score_runtime_window,
                    None,
                )
            };
            if score_runtime_window {
                primary_replica_transport_device
                    .set_scheduler_kernel_timestamp_collection_enabled(false);
                for replica_device in &replica_transport_devices {
                    replica_device.set_scheduler_kernel_timestamp_collection_enabled(false);
                }
            }
            let result = result.with_context(|| {
                format!(
                    "training budgeted Vulkan epoch {} optimizer window starting at batch {}",
                    epoch + 1,
                    batch_index + 1
                )
            })?;
            let runtime_elapsed_seconds = runtime_window_started.elapsed().as_secs_f64();
            let batch_tokens = u64::try_from(args.batch_size)
                .context("native batch size exceeds u64 for runtime telemetry")?;
            let runtime_total_tokens = labeled_inputs.iter().try_fold(0u64, |total, input| {
                let sequence_tokens = u64::try_from(input.tokens)
                    .context("sequence token count exceeds u64 for runtime telemetry")?;
                total
                    .checked_add(sequence_tokens.saturating_mul(batch_tokens))
                    .context("native runtime token counter overflow")
            })?;
            let window_mean_loss = finite_mean(&result.losses);
            if let (Some(arm_index), Some(latency_before)) =
                (runtime_arm_index, runtime_latency_before)
            {
                let latency_after = graph
                    .submission_arena_stats()
                    .context("snapshotting Vulkan submission latency after joint runtime arm")?;
                let mut latency = SubmissionLatencyDelta::between(latency_before, latency_after);
                // Separate Vulkan logical devices own disjoint timestamp/timeline
                // arenas, so fold their deltas into the arm-level profile. A
                // device-group replica shares the primary arena and is already
                // covered by the outer primary snapshot; skip it here to avoid
                // double-counting the same query/timeline samples.
                for device_profile in result
                    .runtime_device_profiles
                    .iter()
                    .filter(|profile| profile.lane_index != 0 && !profile.shared_submission_arena)
                {
                    latency.samples = latency
                        .samples
                        .saturating_add(device_profile.latency.samples);
                    latency.total_ns = latency
                        .total_ns
                        .saturating_add(device_profile.latency.total_ns);
                    latency.kernel_profile_samples = latency
                        .kernel_profile_samples
                        .saturating_add(device_profile.latency.kernel_profile_samples);
                    latency.kernel_dispatches = latency
                        .kernel_dispatches
                        .saturating_add(device_profile.latency.kernel_dispatches);
                    latency.kernel_gpu_ns_total = latency
                        .kernel_gpu_ns_total
                        .saturating_add(device_profile.latency.kernel_gpu_ns_total);
                }
                if let Some(autotuner) = joint_runtime_autotuner.as_mut() {
                    autotuner.observe_with_phase(
                        arm_index,
                        runtime_total_tokens,
                        runtime_elapsed_seconds,
                        result.queue_submissions,
                        latency,
                        &result.runtime_device_profiles,
                        result.runtime_phase_profile,
                    );
                    joint_runtime_profile_scored_windows_since_persist =
                        joint_runtime_profile_scored_windows_since_persist.saturating_add(1);
                    let live_profile = joint_runtime_profile_identity
                        .as_ref()
                        .and_then(|profile_key| autotuner.persisted_profile(profile_key.clone()));
                    if let Some(profile) = live_profile {
                        let winner_changed =
                            last_persisted_joint_runtime_winner != Some(profile.winning_arm);
                        let persist_due = winner_changed
                            || joint_runtime_profile_scored_windows_since_persist
                                >= JOINT_RUNTIME_PROFILE_LIVE_PERSIST_EVERY_SCORED_WINDOWS;
                        if persist_due {
                            let winning_arm = profile.winning_arm;
                            match write_joint_runtime_profile(&args.output_dir, &profile) {
                                Ok(path) => {
                                    joint_runtime_profile_scored_windows_since_persist = 0;
                                    last_persisted_joint_runtime_winner = Some(winning_arm);
                                    if std::env::var_os(HIERARCHOS_JOINT_RUNTIME_AUTOTUNE_LOG_ENV)
                                        .is_some()
                                    {
                                        eprintln!(
                                            "vulkan_joint_runtime_profile persist=live-ok path={} winning_transport_chunk={} winning_tape_microbatch={} winning_checkpoint_stride={} winning_optimizer_broadcast_overlap={}",
                                            path.display(),
                                            winning_arm.gradient_stream_chunk_values,
                                            winning_arm.tape_geometry.sequence_microbatch_size,
                                            winning_arm.tape_geometry.state_checkpoint_stride,
                                            winning_arm.optimizer_broadcast_overlap,
                                        );
                                    }
                                }
                                Err(err) => {
                                    joint_runtime_profile_scored_windows_since_persist = 0;
                                    eprintln!(
                                        "vulkan_joint_runtime_profile persist=live-failed path={} reason={err:#}",
                                        args.output_dir
                                            .join(JOINT_RUNTIME_PROFILE_FILENAME)
                                            .display()
                                    );
                                }
                            }
                        }
                    }
                }
            }
            total_queue_submissions = total_queue_submissions
                .checked_add(result.queue_submissions)
                .context("native queue-submission counter overflow")?;
            total_optimizer_wavefront_windows = total_optimizer_wavefront_windows
                .checked_add(result.optimizer_wavefront_windows as u64)
                .context("native optimizer wavefront-window counter overflow")?;
            total_optimizer_wavefront_ranges = total_optimizer_wavefront_ranges
                .checked_add(result.optimizer_wavefront_ranges as u64)
                .context("native optimizer wavefront-range counter overflow")?;
            total_gradient_reductions = total_gradient_reductions
                .checked_add(result.gradient_reductions as u64)
                .context("native gradient-reduction counter overflow")?;
            total_replica_broadcasts = total_replica_broadcasts
                .checked_add(result.replica_broadcasts as u64)
                .context("native replica-broadcast counter overflow")?;
            total_replica_broadcast_compute_handoffs = total_replica_broadcast_compute_handoffs
                .checked_add(result.replica_broadcast_compute_handoffs as u64)
                .context("native replica broadcast-to-compute handoff counter overflow")?;
            total_gradient_stream_chunks = total_gradient_stream_chunks
                .checked_add(result.gradient_stream_chunks as u64)
                .context("native gradient-stream chunk counter overflow")?;
            total_gradient_stream_values = total_gradient_stream_values
                .checked_add(result.gradient_stream_values as u64)
                .context("native gradient-stream value counter overflow")?;
            max_gradient_stream_pipeline_slots =
                max_gradient_stream_pipeline_slots.max(result.gradient_stream_pipeline_slots);
            total_gradient_stream_persistent_reuses = total_gradient_stream_persistent_reuses
                .checked_add(result.gradient_stream_persistent_reuses as u64)
                .context("native persistent transport reuse counter overflow")?;
            peak_gradient_stream_host_bytes =
                peak_gradient_stream_host_bytes.max(result.gradient_stream_peak_host_bytes);
            peak_gradient_stream_device_bytes =
                peak_gradient_stream_device_bytes.max(result.gradient_stream_peak_device_bytes);
            peak_gradient_stream_host_heap_bytes = peak_gradient_stream_host_heap_bytes
                .max(result.gradient_stream_peak_host_heap_bytes);
            observed_gradient_stream_backends.extend(result.gradient_stream_backends);
            total_replica_state_stream_chunks = total_replica_state_stream_chunks
                .checked_add(result.replica_state_stream_chunks as u64)
                .context("native replica-state stream chunk counter overflow")?;
            total_replica_state_stream_values = total_replica_state_stream_values
                .checked_add(result.replica_state_stream_values as u64)
                .context("native replica-state stream value counter overflow")?;
            max_replica_state_stream_pipeline_slots = max_replica_state_stream_pipeline_slots
                .max(result.replica_state_stream_pipeline_slots);
            total_replica_state_stream_persistent_reuses =
                total_replica_state_stream_persistent_reuses
                    .checked_add(result.replica_state_stream_persistent_reuses as u64)
                    .context("native replica-state persistent transport reuse counter overflow")?;
            peak_replica_state_stream_host_bytes = peak_replica_state_stream_host_bytes
                .max(result.replica_state_stream_peak_host_bytes);
            peak_replica_state_stream_device_bytes = peak_replica_state_stream_device_bytes
                .max(result.replica_state_stream_peak_device_bytes);
            observed_replica_state_stream_backends.extend(result.replica_state_stream_backends);
            total_replica_state_host_fallbacks = total_replica_state_host_fallbacks
                .checked_add(result.replica_state_host_fallbacks as u64)
                .context("native replica-state host fallback counter overflow")?;
            let previous_optimizer_step = optimizer_step;
            optimizer_step = result.optimizer_step;
            anyhow::ensure!(
                optimizer_step == previous_optimizer_step
                    || optimizer_step == previous_optimizer_step.saturating_add(1),
                "budgeted native optimizer step jumped from {} to {}",
                previous_optimizer_step,
                optimizer_step
            );
            if optimizer_step > previous_optimizer_step {
                advance_lr_schedule(&mut schedule)?;
            } else if loss_scaling.is_some() {
                account_skipped_training_batch(
                    &mut skipped_train_batches,
                    args.max_skipped_train_batches,
                    epoch,
                    batch_index,
                    "dynamic loss scaler rejected a non-finite optimizer window",
                )?;
            }
            for loss in result.losses.into_iter().filter(|value| value.is_finite()) {
                epoch_loss_sum += f64::from(loss);
                epoch_loss_values += 1;
            }
            let completed_batch_number = batch_index + window_batch_count;
            let completed_global_step = epoch
                .saturating_mul(u64::try_from(full_batch_count).unwrap_or(u64::MAX))
                .saturating_add(u64::try_from(completed_batch_number).unwrap_or(u64::MAX));
            emit_gui_event(
                args.json_events,
                serde_json::json!({
                    "event": "training_metrics",
                    "backend": "vulkan",
                    "epoch": epoch + 1,
                    "step": completed_global_step,
                    "total_steps": gui_total_steps,
                    "optimizer_step": optimizer_step,
                    "loss": window_mean_loss.unwrap_or(0.0),
                    "loss_available": window_mean_loss.is_some(),
                    "lr": live_lr,
                    "tokens_per_sec": if runtime_elapsed_seconds > 0.0 {
                        runtime_total_tokens as f64 / runtime_elapsed_seconds
                    } else {
                        0.0
                    },
                }),
            );
            let checkpoint_due =
                periodic_checkpoint_crossed(batch_index, completed_batch_number, args.save_steps);
            if checkpoint_due {
                total_queue_submissions = total_queue_submissions
                    .checked_add(u64::from(
                        graph
                            .synchronize_ltm_alignment_controller_metadata()
                            .context("synchronizing Vulkan LTM checkpoint metadata")?,
                    ))
                    .context("periodic checkpoint metadata submission counter overflow")?;
            }
            cursor = window_cursor;
            if checkpoint_due {
                let execution_policy =
                    training_execution_policy(training_precision, loss_scaling.as_ref());
                let replay = portable_training_replay(
                    &graph,
                    &args,
                    &cursor,
                    &schedule,
                    skipped_train_batches,
                    execution_policy,
                    carried_h_state.as_deref(),
                    carried_l_state.as_deref(),
                    &previous_context,
                    &target_context,
                    portable_ltm_state.as_ref(),
                    &run_identity,
                )?;
                let checkpoint_dir =
                    periodic_checkpoint_dir(&args.output_dir, epoch, completed_batch_number);
                graph
                    .export_training_checkpoint_package_with_replay(
                        &args.model_dir,
                        &checkpoint_dir,
                        &replay,
                    )
                    .with_context(|| {
                        format!(
                            "exporting periodic cross-backend checkpoint to {}",
                            checkpoint_dir.display()
                        )
                    })?;
                eprintln!(
                    "vulkan_periodic_checkpoint epoch={} step={} path={} boundary=optimizer-window",
                    epoch + 1,
                    completed_batch_number,
                    checkpoint_dir.display()
                );
            }
            if cursor.epoch != epoch {
                let mean_loss = if epoch_loss_values == 0 {
                    f64::NAN
                } else {
                    epoch_loss_sum / epoch_loss_values as f64
                };
                eprintln!(
                    "epoch {}/{}: batches={} optimizer_step={} live_lr={:.8e} mean_recorded_loss={:.6}",
                    epoch + 1,
                    args.epochs,
                    full_batch_count,
                    optimizer_step,
                    schedule_live_lr(&schedule, args.learning_rate)?,
                    mean_loss,
                );
                epoch_loss_sum = 0.0;
                epoch_loss_values = 0;
                carried_h_state = None;
                carried_l_state = None;
                previous_context.fill(0.0);
                target_context.fill(0.0);
            }
            continue;
        }
        let batch_indices = cursor
            .current_batch_indices(Some(&dataset_lengths))?
            .context("native data cursor has no runnable batch")?;
        anyhow::ensure!(
            batch_indices.len() == args.batch_size,
            "drop-last native trainer produced a partial batch"
        );
        let batch_rows = batch_indices
            .iter()
            .map(|&index| &dataset[index])
            .collect::<Vec<_>>();
        let packed = pack_batch(&batch_rows, config.vocab_size)?;
        let mut tape = if args.persist_state {
            match (carried_h_state.as_deref(), carried_l_state.as_deref()) {
                (Some(h_state), Some(l_state)) => graph
                    .create_token_tape(args.batch_size, packed.tokens, h_state, l_state)
                    .context("creating continuation Vulkan token tape")?,
                (None, None) => graph
                    .create_zero_token_tape(args.batch_size, packed.tokens)
                    .context("creating fresh Vulkan token tape")?,
                _ => bail!("native persistent H/L carrier became internally inconsistent"),
            }
        } else {
            graph
                .create_zero_token_tape(args.batch_size, packed.tokens)
                .context("creating zero-state Vulkan token tape")?
        };
        let live_lr = schedule_live_lr(&schedule, args.learning_rate)?;
        let hyper = AdamWHyperParams {
            lr: live_lr,
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            weight_decay: args.weight_decay,
        };
        let input = HierarchosRawTokenLabeledSequenceInput {
            tokens: packed.tokens,
            input_ids: &packed.input_ids,
            labels: &packed.labels,
            attention_mask: Some(&packed.attention_mask),
            loss_weights: Some(&packed.loss_weights),
            initial_previous_context: &previous_context,
            initial_target_context: &target_context,
            // PyTorch's persisted recurrent carrier crosses batches, but its
            // absolute position is local to the current batch and advances only
            // across TBPTT chunks inside that batch.
            global_pos_offset: 0,
            reset_rosa_at_start: !args.persist_state || cursor.batch_cursor == 0,
            pytorch_tbptt_chunk_size: args.tbptt_chunk_size,
        };
        let optimizer_boundary = optimizer_window_ends(
            batch_index,
            full_batch_count,
            args.gradient_accumulation_steps,
        );
        let dynamic_window_end = loss_scaling.is_some() && optimizer_boundary;
        let batch_started = Instant::now();
        let result = if let Some(loss_scaling) = loss_scaling.as_mut() {
            let accumulation_mode =
                dynamic_accumulation_mode(batch_index, args.gradient_accumulation_steps);
            if dynamic_loss_scaling_device_resident {
                graph.train_raw_token_labeled_sequence_with_device_resident_dynamic_loss_scaling(
                    &mut tape,
                    &input,
                    objective,
                    hyper,
                    accumulation_mode,
                    loss_scaling,
                )
            } else {
                graph.train_raw_token_labeled_sequence_with_dynamic_loss_scaling(
                    &mut tape,
                    &input,
                    objective,
                    hyper,
                    accumulation_mode,
                    loss_scaling,
                )
            }
        } else {
            graph.train_raw_token_labeled_sequence_with_update_mode(
                &mut tape,
                &input,
                objective,
                hyper,
                update_mode(
                    batch_index,
                    full_batch_count,
                    args.gradient_accumulation_steps,
                ),
            )
        }
        .with_context(|| {
            format!(
                "training Vulkan epoch {} batch {}",
                epoch + 1,
                batch_index + 1
            )
        })?;
        let batch_elapsed_seconds = batch_started.elapsed().as_secs_f64();
        let batch_mean_loss = finite_mean(&result.losses);
        total_queue_submissions = total_queue_submissions
            .checked_add(u64::from(result.queue_submissions))
            .context("native per-batch queue-submission counter overflow")?;
        let previous_optimizer_step = optimizer_step;
        optimizer_step = if dynamic_window_end {
            let finish = graph
                .finish_full_model_accumulation_with_dynamic_loss_scaling_and_gradient_clipping_wavefront_device_resident(
                    hyper,
                    loss_scaling
                        .as_mut()
                        .context("dynamic optimizer window lost its GradScaler state")?,
                    args.grad_clip,
                    args.gradient_stream_chunk_values,
                )
                .context("closing queue-resident Vulkan GradScaler optimizer window")?;
            let (decision, observation_submissions) = graph
                .observe_dynamic_loss_scaling_decision_preserving_device_authority_with_submission_count()
                .context("observing queue-resident Vulkan GradScaler decision")?;
            let loss_scaling = loss_scaling
                .as_mut()
                .context("dynamic optimizer window lost its GradScaler state")?;
            // Keep the host copy as scheduler/control-plane telemetry only. The
            // next backward consumes the live Vulkan scale directly and does
            // not transfer GradScaler authority back to Rust.
            loss_scaling.scale = Some(decision.scale_after);
            loss_scaling.growth_tracker = Some(decision.growth_tracker);
            loss_scaling.pending_gradients_scaled = false;
            dynamic_loss_scaling_device_resident = true;
            total_queue_submissions = total_queue_submissions
                .checked_add(u64::from(finish.queue_submissions))
                .and_then(|count| count.checked_add(u64::from(observation_submissions)))
                .context("native dynamic optimizer submission counter overflow")?;
            if decision.should_step {
                previous_optimizer_step
                    .checked_add(1)
                    .context("native queue-resident dynamic optimizer-step counter overflow")?
            } else {
                previous_optimizer_step
            }
        } else if loss_scaling.is_some() {
            // Device-resident AdamW clocks intentionally remain stale on the
            // host between optimizer boundaries. The scheduler advances this
            // scalar only after the queue-resident GradScaler step decision.
            previous_optimizer_step
        } else {
            result.full_model_optimizer.step
        };
        anyhow::ensure!(
            optimizer_step == previous_optimizer_step
                || optimizer_step == previous_optimizer_step.saturating_add(1),
            "native optimizer step jumped from {} to {}",
            previous_optimizer_step,
            optimizer_step
        );
        if optimizer_step > previous_optimizer_step {
            advance_lr_schedule(&mut schedule)?;
        } else if dynamic_window_end {
            account_skipped_training_batch(
                &mut skipped_train_batches,
                args.max_skipped_train_batches,
                epoch,
                batch_index,
                "dynamic loss scaler rejected a non-finite optimizer window",
            )?;
        }
        if args.save_steps > 0 && optimizer_boundary {
            total_queue_submissions = total_queue_submissions
                .checked_add(u64::from(
                    graph
                        .synchronize_ltm_alignment_controller_metadata()
                        .context("synchronizing Vulkan LTM checkpoint metadata")?,
                ))
                .context("periodic checkpoint metadata submission counter overflow")?;
        }
        for loss in result.losses.into_iter().filter(|value| value.is_finite()) {
            epoch_loss_sum += f64::from(loss);
            epoch_loss_values += 1;
        }
        if args.persist_state {
            carried_h_state = Some(result.final_h_packed_state);
            carried_l_state = Some(result.final_l_packed_state);
            previous_context = result
                .final_previous_context
                .context("labeled Vulkan tape did not return final previous_context")?;
            target_context = result
                .final_target_context
                .context("labeled Vulkan tape did not return final target_context")?;
        }

        let rolled_epoch = cursor.advance_batch()?;
        let completed_batch_number = batch_index + 1;
        let completed_global_step = epoch
            .saturating_mul(u64::try_from(full_batch_count).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(completed_batch_number).unwrap_or(u64::MAX));
        let batch_total_tokens = packed.tokens.saturating_mul(args.batch_size);
        emit_gui_event(
            args.json_events,
            serde_json::json!({
                "event": "training_metrics",
                "backend": "vulkan",
                "epoch": epoch + 1,
                "step": completed_global_step,
                "total_steps": gui_total_steps,
                "optimizer_step": optimizer_step,
                "loss": batch_mean_loss.unwrap_or(0.0),
                "loss_available": batch_mean_loss.is_some(),
                "lr": live_lr,
                "tokens_per_sec": if batch_elapsed_seconds > 0.0 {
                    batch_total_tokens as f64 / batch_elapsed_seconds
                } else {
                    0.0
                },
            }),
        );
        if periodic_checkpoint_due(batch_index + 1, args.save_steps) {
            let execution_policy =
                training_execution_policy(training_precision, loss_scaling.as_ref());
            let replay = portable_training_replay(
                &graph,
                &args,
                &cursor,
                &schedule,
                skipped_train_batches,
                execution_policy,
                carried_h_state.as_deref(),
                carried_l_state.as_deref(),
                &previous_context,
                &target_context,
                portable_ltm_state.as_ref(),
                &run_identity,
            )?;
            let checkpoint_dir = periodic_checkpoint_dir(&args.output_dir, epoch, batch_index + 1);
            graph
                .export_training_checkpoint_package_with_replay(
                    &args.model_dir,
                    &checkpoint_dir,
                    &replay,
                )
                .with_context(|| {
                    format!(
                        "exporting periodic cross-backend checkpoint to {}",
                        checkpoint_dir.display()
                    )
                })?;
            eprintln!(
                "vulkan_periodic_checkpoint epoch={} step={} path={}",
                epoch + 1,
                batch_index + 1,
                checkpoint_dir.display()
            );
        }
        if rolled_epoch {
            let mean_loss = if epoch_loss_values == 0 {
                f64::NAN
            } else {
                epoch_loss_sum / epoch_loss_values as f64
            };
            eprintln!(
                "epoch {}/{}: batches={} optimizer_step={} live_lr={:.8e} mean_recorded_loss={:.6}",
                epoch + 1,
                args.epochs,
                full_batch_count,
                optimizer_step,
                schedule_live_lr(&schedule, args.learning_rate)?,
                mean_loss,
            );
            epoch_loss_sum = 0.0;
            epoch_loss_values = 0;
            carried_h_state = None;
            carried_l_state = None;
            previous_context.fill(0.0);
            target_context.fill(0.0);
        }
    }

    // The last optimizer window intentionally leaves its replica broadcast in
    // flight. Retire it before checkpoint/export teardown so every worker has
    // consumed the final generation and transport telemetry is accounted for.
    let final_broadcast =
        retire_replica_state_broadcast(&mut pending_replica_broadcast, replica_count)?;
    total_queue_submissions = total_queue_submissions
        .checked_add(final_broadcast.queue_submissions)
        .context("final replica-state submission counter overflow")?;
    total_replica_state_stream_chunks = total_replica_state_stream_chunks
        .checked_add(final_broadcast.stream_chunks as u64)
        .context("final replica-state stream chunk counter overflow")?;
    total_replica_state_stream_values = total_replica_state_stream_values
        .checked_add(final_broadcast.stream_values as u64)
        .context("final replica-state stream value counter overflow")?;
    max_replica_state_stream_pipeline_slots =
        max_replica_state_stream_pipeline_slots.max(final_broadcast.stream_pipeline_slots);
    total_replica_state_stream_persistent_reuses = total_replica_state_stream_persistent_reuses
        .checked_add(final_broadcast.stream_persistent_reuses as u64)
        .context("final replica-state persistent transport reuse counter overflow")?;
    peak_replica_state_stream_host_bytes =
        peak_replica_state_stream_host_bytes.max(final_broadcast.stream_peak_host_bytes);
    peak_replica_state_stream_device_bytes =
        peak_replica_state_stream_device_bytes.max(final_broadcast.stream_peak_device_bytes);
    observed_replica_state_stream_backends.extend(final_broadcast.stream_backends);
    total_replica_state_host_fallbacks = total_replica_state_host_fallbacks
        .checked_add(final_broadcast.host_fallbacks as u64)
        .context("final replica-state host fallback counter overflow")?;

    anyhow::ensure!(
        cursor.batch_cursor == 0,
        "native trainer reached target epochs away from an epoch boundary"
    );
    if dynamic_loss_scaling_device_resident {
        let synchronized = graph
            .synchronize_dynamic_loss_scaling_metadata(
                loss_scaling
                    .as_mut()
                    .context("queue-resident dynamic training lost its GradScaler state")?,
            )
            .context("synchronizing queue-resident GradScaler/AdamW checkpoint metadata")?;
        anyhow::ensure!(
            synchronized.full_model_optimizer.step == optimizer_step,
            "queue-resident optimizer step {} disagrees with synchronized Vulkan AdamW step {}",
            optimizer_step,
            synchronized.full_model_optimizer.step,
        );
        total_queue_submissions = total_queue_submissions
            .checked_add(u64::from(synchronized.queue_submissions))
            .context("final dynamic metadata synchronization submission counter overflow")?;
    }
    // FP32 and storage-only training can also leave the LTM writer-readiness
    // controller device-authoritative after the last committed optimizer step.
    // Periodic checkpoints synchronize it opportunistically, but the final
    // export must be correct even when --save-steps=0. Dynamic-loss-scaling
    // synchronization above already includes this readback, so the idempotent
    // call is normally a zero-submission no-op on that path.
    total_queue_submissions = total_queue_submissions
        .checked_add(u64::from(
            graph
                .synchronize_ltm_alignment_controller_metadata()
                .context("synchronizing final Vulkan LTM checkpoint metadata")?,
        ))
        .context("final LTM metadata synchronization submission counter overflow")?;
    let execution_policy = training_execution_policy(training_precision, loss_scaling.as_ref());
    let replay = portable_training_replay(
        &graph,
        &args,
        &cursor,
        &schedule,
        skipped_train_batches,
        execution_policy.clone(),
        carried_h_state.as_deref(),
        carried_l_state.as_deref(),
        &previous_context,
        &target_context,
        portable_ltm_state.as_ref(),
        &run_identity,
    )?;
    let manifest = graph
        .export_training_checkpoint_package_with_replay(&args.model_dir, &args.output_dir, &replay)
        .with_context(|| {
            format!(
                "exporting cross-backend training package to {}",
                args.output_dir.display()
            )
        })?;
    let persistent_joint_runtime_profile =
        joint_runtime_profile_identity
            .clone()
            .and_then(|profile_key| {
                joint_runtime_autotuner
                    .as_ref()
                    .and_then(|autotuner| autotuner.persisted_profile(profile_key.clone()))
                    .or_else(|| {
                        loaded_joint_runtime_profile
                            .as_ref()
                            .filter(|profile| profile.profile_key == profile_key)
                            .cloned()
                    })
            });
    let persistent_joint_runtime_profile_path = if let Some(profile) =
        persistent_joint_runtime_profile.as_ref()
    {
        match write_joint_runtime_profile(&args.output_dir, profile) {
            Ok(path) => {
                eprintln!(
                        "vulkan_joint_runtime_profile persist=ok path={} winning_transport_chunk={} winning_tape_microbatch={} winning_checkpoint_stride={} winning_optimizer_broadcast_overlap={}",
                        path.display(),
                        profile.winning_arm.gradient_stream_chunk_values,
                        profile.winning_arm.tape_geometry.sequence_microbatch_size,
                        profile.winning_arm.tape_geometry.state_checkpoint_stride,
                        profile.winning_arm.optimizer_broadcast_overlap,
                    );
                Some(path)
            }
            Err(err) => {
                eprintln!(
                    "vulkan_joint_runtime_profile persist=failed path={} reason={err:#}",
                    args.output_dir
                        .join(JOINT_RUNTIME_PROFILE_FILENAME)
                        .display()
                );
                None
            }
        }
    } else {
        None
    };
    let reported_device_indices = args
        .device_indices
        .clone()
        .unwrap_or_else(|| vec![primary_device_index]);
    let runtime_gradient_transport = if replica_count == 0 {
        "single-device"
    } else if observed_gradient_stream_backends.len() == 1 {
        match observed_gradient_stream_backends.iter().next().copied() {
            Some(VulkanGradientTransportBackend::DeviceGroup) => {
                if grouped_device_views
                    .as_ref()
                    .is_some_and(|views| views[0].device_group_timeline_semaphore_enabled())
                {
                    "device-group-peer-memory-v3-timeline"
                } else {
                    "device-group-peer-memory-v2-semaphore"
                }
            }
            Some(VulkanGradientTransportBackend::OpaqueExternalMemory) => {
                "opaque-external-memory-v1-binary-semaphore"
            }
            Some(VulkanGradientTransportBackend::HostVisibleStagedV2Pipelined) => {
                "host-visible-staged-v2-pipelined"
            }
            None => "host-visible-staged-v2-pipelined",
        }
    } else if observed_gradient_stream_backends.len() > 1 {
        "mixed-vulkan-gradient-transports"
    } else {
        "host-visible-staged-v2-pipelined"
    };
    let runtime_replica_state_transport = if replica_count == 0 {
        "single-device"
    } else if total_replica_state_host_fallbacks > 0
        && !observed_replica_state_stream_backends.is_empty()
    {
        "mixed-vulkan-direct-and-host-snapshot"
    } else if total_replica_state_host_fallbacks > 0 {
        "portable-host-snapshot"
    } else if observed_replica_state_stream_backends.len() == 1 {
        observed_replica_state_stream_backends
            .iter()
            .next()
            .copied()
            .map(VulkanGradientTransportBackend::label)
            .unwrap_or("portable-host-snapshot")
    } else if observed_replica_state_stream_backends.len() > 1 {
        "mixed-vulkan-direct"
    } else {
        "portable-host-snapshot"
    };
    let device_group_queue_lanes = grouped_device_views
        .as_ref()
        .map(|views| views[0].queue_lane_count())
        .unwrap_or(1);
    let device_group_timeline_semaphore = grouped_device_views
        .as_ref()
        .is_some_and(|views| views[0].device_group_timeline_semaphore_enabled());
    let primary_device_info = device_catalog
        .iter()
        .find(|device| device.index == reported_device_indices[0])
        .context("primary Vulkan device disappeared from transport capability catalog")?;
    let gradient_transport_pairs = reported_device_indices[1..]
        .iter()
        .map(|&replica_index| -> Result<serde_json::Value> {
            let replica = device_catalog
                .iter()
                .find(|device| device.index == replica_index)
                .with_context(|| {
                    format!(
                        "Vulkan replica device {replica_index} disappeared from transport capability catalog"
                    )
                })?;
            let device_group_candidate =
                primary_device_info.device_group_transport_candidate_with(replica);
            let opaque_external_memory_candidate = primary_device_info
                .opaque_external_memory_transport_candidate_with(replica);
            let opaque_external_semaphore_candidate = primary_device_info
                .external_semaphore
                .platform_bidirectional_candidate()
                && replica
                    .external_semaphore
                    .platform_bidirectional_candidate();
            let opaque_external_transport_candidate = primary_device_info
                .opaque_external_transport_candidate_with(replica);
            let transport_plan = primary_device_info.gradient_transport_plan_with(replica);
            Ok(serde_json::json!({
                "primary_device_index": primary_device_info.index,
                "replica_device_index": replica.index,
                "primary_device_uuid": primary_device_info.device_uuid,
                "replica_device_uuid": replica.device_uuid,
                "active_route": if device_group_candidate && runtime_gradient_transport.starts_with("device-group-peer-memory-") {
                    runtime_gradient_transport
                } else if opaque_external_probed_pairs.contains(&replica.index) && runtime_gradient_transport.starts_with("opaque-external-memory-") {
                    runtime_gradient_transport
                } else {
                    transport_plan.active_backend.label()
                },
                "device_group_candidate": device_group_candidate,
                "device_group_queue_lanes": device_group_queue_lanes,
                "device_group_timeline_semaphore": device_group_timeline_semaphore,
                "opaque_external_memory_candidate": opaque_external_memory_candidate,
                "opaque_external_semaphore_candidate": opaque_external_semaphore_candidate,
                "opaque_external_transport_candidate": opaque_external_transport_candidate,
                "opaque_external_transport_probe_passed": opaque_external_probed_pairs.contains(&replica.index),
                "opaque_external_memory_handle": if opaque_external_memory_candidate {
                    primary_device_info.external_buffer.platform_handle_name()
                } else {
                    None
                },
                "opaque_external_semaphore_handle": if opaque_external_semaphore_candidate {
                    primary_device_info.external_semaphore.platform_handle_name()
                } else {
                    None
                },
                "future_preferred_route": transport_plan
                    .direct_candidate
                    .map(|backend| backend.label())
                    .unwrap_or_else(|| transport_plan.active_backend.label()),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let replica_state_stream_report = serde_json::json!({
        "chunks": total_replica_state_stream_chunks,
        "values": total_replica_state_stream_values,
        "pipeline_slots": max_replica_state_stream_pipeline_slots,
        "persistent_reuses": total_replica_state_stream_persistent_reuses,
        "broadcast_scheduler": "resident-replica-workers-v9-device-timeline-resource-arena",
        "transient_submission_retirement": "device-timeline-resource-arena",
        "persistent_device_group_slots": "gpu-dependency-only",
        "worker_threads": replica_count,
        "worker_jobs": total_replica_broadcasts,
        "worker_reuses": total_replica_broadcasts.saturating_sub(replica_count as u64),
        "compute_handoffs": total_replica_broadcast_compute_handoffs,
        "per_replica_compute_handoff": replica_count > 0,
        "resident_replica_graphs": replica_count > 0,
        "primary_compute_overlap_enabled": replica_count > 0,
        "ordered_reduction_overlap_enabled": replica_count > 0,
        "optimizer_generation_guard": "chunk-range-retirement-v3-predeclared-device-group-timeline",
        "optimizer_wavefront_enabled": replica_count > 0,
        "optimizer_wavefront_submission": "async-primary-queue-tail-drain",
        "optimizer_wavefront_per_run_host_fence_wait": false,
        "optimizer_device_group_timeline_waits": replica_count > 0 && device_group_timeline_semaphore,
        "optimizer_predeclared_retirement_timeline_lanes": predeclared_retirement_timeline_lanes,
        "optimizer_predeclared_retirement_timeline_enabled": predeclared_retirement_timeline_lanes > 0,
        "optimizer_wavefront_windows": total_optimizer_wavefront_windows,
        "optimizer_wavefront_ranges": total_optimizer_wavefront_ranges,
        "optimizer_wavefront_chunk_values": args.gradient_stream_chunk_values,
        "gradient_reduction_order": "replica-index",
        "peak_host_bytes": peak_replica_state_stream_host_bytes,
        "peak_device_bytes": peak_replica_state_stream_device_bytes,
        "host_fallbacks": total_replica_state_host_fallbacks,
    });
    let joint_runtime_autotune_report = joint_runtime_autotuner
        .as_ref()
        .map(JointRuntimeAutotuner::report);
    let mut training_report = serde_json::json!({
        "backend": "vulkan",
        "model_abi": "canonical-safetensors-fp32-master",
        "device_indices": reported_device_indices,
        "data_parallel_replicas": replica_count + 1,
        "gradient_transport": runtime_gradient_transport,
        "replica_state_transport": runtime_replica_state_transport,
        "gradient_transport_pairs": gradient_transport_pairs,
        "device_group_queue_lanes": device_group_queue_lanes,
        "device_group_timeline_semaphore": device_group_timeline_semaphore,
        "dataset_rows": dataset.len(),
        "dropped_rows_per_epoch": dropped_rows,
        "start_epoch": start_epoch,
        "completed_epochs": cursor.epoch,
        "target_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer_step": optimizer_step,
        "scheduler_step": schedule.step,
        "live_lr": schedule_live_lr(&schedule, args.learning_rate)?,
        "persist_state": args.persist_state,
        "queue_submissions": total_queue_submissions,
        "optimizer_wavefront_windows": total_optimizer_wavefront_windows,
        "optimizer_wavefront_ranges": total_optimizer_wavefront_ranges,
        "gradient_reductions": total_gradient_reductions,
        "replica_broadcasts": total_replica_broadcasts,
        "gradient_stream_chunks": total_gradient_stream_chunks,
        "gradient_stream_values": total_gradient_stream_values,
        "gradient_stream_pipeline_slots": max_gradient_stream_pipeline_slots,
        "gradient_stream_persistent_reuses": total_gradient_stream_persistent_reuses,
        "gradient_stream_peak_host_bytes": peak_gradient_stream_host_bytes,
        "gradient_stream_peak_device_bytes": peak_gradient_stream_device_bytes,
        "gradient_stream_peak_host_heap_bytes": peak_gradient_stream_host_heap_bytes,
        "gradient_stream_chunk_values": args.gradient_stream_chunk_values,
        "gradient_stream_chunk_bytes": args.gradient_stream_chunk_values * std::mem::size_of::<f32>(),
        "gradient_stream_source_lifetime": "detached-vulkan-buffer-clones-v1",
        "phase_aware_joint_memory_plan": joint_memory_preflight,
        "replica_state_stream": replica_state_stream_report,
        "resumed_optimizer": resumed_optimizer,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "resumed_training_session": resumed_session.is_some(),
        "training_precision_policy": manifest.training_precision_policy,
        "execution_policy": execution_policy,
        "dynamic_loss_scaling": dynamic_loss_scaling_enabled,
        "initial_loss_scale_requested": args.initial_loss_scale,
        "training_precision_cli_override": args.training_precision_override,
        "skipped_train_batches": skipped_train_batches,
        "output": args.output_dir,
    });
    let training_report_object = training_report
        .as_object_mut()
        .context("native Vulkan training report must be a JSON object")?;
    training_report_object.insert(
        "joint_runtime_profile_schema_version".to_owned(),
        serde_json::Value::from(JOINT_RUNTIME_PROFILE_SCHEMA_VERSION),
    );
    training_report_object.insert(
        "joint_runtime_profile_filename".to_owned(),
        serde_json::Value::from(JOINT_RUNTIME_PROFILE_FILENAME),
    );
    training_report_object.insert(
        "joint_runtime_profile_loaded".to_owned(),
        serde_json::Value::from(loaded_joint_runtime_profile.is_some()),
    );
    training_report_object.insert(
        "joint_runtime_profile_requested_source".to_owned(),
        args.joint_runtime_profile
            .as_ref()
            .map(|path| serde_json::Value::from(path.display().to_string()))
            .unwrap_or(serde_json::Value::Null),
    );
    training_report_object.insert(
        "joint_runtime_profile_locked".to_owned(),
        serde_json::Value::from(args.lock_joint_runtime_profile),
    );
    training_report_object.insert(
        "joint_runtime_locked_arm".to_owned(),
        joint_runtime_autotuner
            .as_ref()
            .and_then(JointRuntimeAutotuner::locked_arm)
            .map(|arm| serde_json::to_value(arm).expect("joint runtime arm must serialize"))
            .unwrap_or(serde_json::Value::Null),
    );
    training_report_object.insert(
        "joint_runtime_profile_loaded_source".to_owned(),
        loaded_joint_runtime_profile_source
            .as_ref()
            .map(|path| serde_json::Value::from(path.display().to_string()))
            .unwrap_or(serde_json::Value::Null),
    );
    training_report_object.insert(
        "joint_runtime_profile_live_persist_every_scored_windows".to_owned(),
        serde_json::Value::from(JOINT_RUNTIME_PROFILE_LIVE_PERSIST_EVERY_SCORED_WINDOWS),
    );
    training_report_object.insert(
        "joint_runtime_profile_transport_ceiling".to_owned(),
        persistent_profile_transport_ceiling
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    training_report_object.insert(
        "joint_runtime_profile_persisted_path".to_owned(),
        persistent_joint_runtime_profile_path
            .as_ref()
            .map(|path| serde_json::Value::from(path.display().to_string()))
            .unwrap_or(serde_json::Value::Null),
    );
    training_report_object.insert(
        "joint_runtime_profile_checkpoint_abi".to_owned(),
        serde_json::Value::from("runtime-only-sidecar-canonical-fp32-checkpoint-unchanged"),
    );
    training_report_object.insert(
        "joint_runtime_autotune".to_owned(),
        joint_runtime_autotune_report.unwrap_or(serde_json::Value::Null),
    );
    println!("{}", serde_json::to_string_pretty(&training_report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_default_tbptt_chunk_size_matches_root_cli_revision_contract() {
        assert_eq!(
            architecture_default_tbptt_chunk_size("coherent-v9"),
            COHERENT_V9_DEFAULT_TBPTT_CHUNK_SIZE
        );
        assert_eq!(
            architecture_default_tbptt_chunk_size("COHERENT-V9"),
            COHERENT_V9_DEFAULT_TBPTT_CHUNK_SIZE
        );
        assert_eq!(
            architecture_default_tbptt_chunk_size("legacy-v8"),
            LEGACY_V8_DEFAULT_TBPTT_CHUNK_SIZE
        );
        assert_eq!(
            architecture_default_tbptt_chunk_size("unknown-future-revision"),
            LEGACY_V8_DEFAULT_TBPTT_CHUNK_SIZE
        );
    }

    fn signed_test_run_identity(starting_lr: f32) -> serde_json::Value {
        let mut identity = serde_json::json!({
            "format": NATIVE_RUN_IDENTITY_FORMAT,
            "version": 1,
            "objective": {"starting_lr": starting_lr},
            "token_cache": {
                "ordered_record_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            }
        });
        let digest = run_identity_digest(&identity).expect("test run identity must hash");
        identity
            .as_object_mut()
            .expect("test run identity must be an object")
            .insert("sha256".to_owned(), serde_json::Value::String(digest));
        identity
    }

    #[test]
    fn native_run_identity_digest_is_stable_across_json_roundtrip() -> Result<()> {
        let identity = signed_test_run_identity(1.0e-4f32);
        let before = run_identity_digest(&identity)?;
        let encoded = serde_json::to_vec(&identity)?;
        let decoded: serde_json::Value = serde_json::from_slice(&encoded)?;
        let after = run_identity_digest(&decoded)?;
        assert_eq!(before, after);
        Ok(())
    }

    #[test]
    fn training_precision_cli_aliases_canonicalize_to_graph_policy_names() -> Result<()> {
        for (raw, expected) in [
            ("fp32", "fp32"),
            ("fp16", "fp16-storage-fp32-compute"),
            ("fp16-parity", "fp16-storage-parity"),
            ("fp16-lm-backward", "fp16-storage-fp16-lm-backward"),
        ] {
            assert_eq!(parse_training_precision(OsString::from(raw))?, expected);
        }
        assert!(parse_training_precision(OsString::from("int8"))
            .expect_err("unsupported training precision must fail closed")
            .to_string()
            .contains("unsupported --precision"));
        Ok(())
    }

    #[test]
    fn fresh_grad_scaler_uses_hardware_qualified_initial_scale() {
        let state = fresh_pytorch_grad_scaler_state(1024.0);
        assert_eq!(state.mode, "dynamic");
        assert_eq!(state.scale, Some(1024.0));
        assert_eq!(state.growth_factor, Some(2.0));
        assert_eq!(state.backoff_factor, Some(0.5));
        assert_eq!(state.growth_interval, Some(2_000));
        assert_eq!(state.growth_tracker, Some(0));
        assert!(!state.pending_gradients_scaled);
    }

    #[test]
    fn schema_v6_token_cache_is_decoded_and_bound_to_python_record_identity() -> Result<()> {
        use safetensors::{serialize_to_file, tensor::TensorView};

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hierarchos-vulkan-portable-token-cache-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root)?;
        let result = (|| -> Result<()> {
            let mut tokens = Vec::new();
            for value in [10u16, 11, 12, 50_000, 10, 11] {
                tokens.extend_from_slice(&value.to_le_bytes());
            }
            for value in [20u16, 21, 50_000, 20] {
                tokens.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(root.join("tokens.bin"), &tokens)?;

            let offsets = [0i64, 12i64];
            let lengths = [3i32, 2i32];
            let loss_run_offsets = [0i64, 2i64, 3i64];
            let loss_run_ends = [2i32, 3i32, 2i32];
            let loss_run_codes = [0u8, 1u8, 1u8];
            let offsets_bytes = offsets
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let lengths_bytes = lengths
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let loss_run_offsets_bytes = loss_run_offsets
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let loss_run_ends_bytes = loss_run_ends
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let index_path = root.join(PORTABLE_TOKEN_CACHE_INDEX_FILENAME);
            serialize_to_file(
                [
                    (
                        "offsets",
                        TensorView::new(Dtype::I64, vec![offsets.len()], &offsets_bytes)?,
                    ),
                    (
                        "lengths",
                        TensorView::new(Dtype::I32, vec![lengths.len()], &lengths_bytes)?,
                    ),
                    (
                        "loss_run_offsets",
                        TensorView::new(
                            Dtype::I64,
                            vec![loss_run_offsets.len()],
                            &loss_run_offsets_bytes,
                        )?,
                    ),
                    (
                        "loss_run_ends",
                        TensorView::new(
                            Dtype::I32,
                            vec![loss_run_ends.len()],
                            &loss_run_ends_bytes,
                        )?,
                    ),
                    (
                        "loss_run_codes",
                        TensorView::new(Dtype::U8, vec![loss_run_codes.len()], &loss_run_codes)?,
                    ),
                ],
                None,
                &index_path,
            )?;

            let mut ordered = Sha256::new();
            ordered.update(TOKEN_CACHE_RECORD_HASH_HEADER);
            ordered.update(3u64.to_le_bytes());
            ordered.update(&tokens[..12]);
            ordered.update([0u8, 0, 1]);
            ordered.update(2u64.to_le_bytes());
            ordered.update(&tokens[12..]);
            ordered.update([1u8, 1]);
            let ordered_sha = format!("{:x}", ordered.finalize());
            let index_sha = sha256_hex(&fs::read(&index_path)?);
            let tokens_sha = sha256_hex(&tokens);
            fs::write(
                root.join("_SUCCESS"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "format": "tokenized-binary-v6-test",
                    "cache_key": "fixture-cache-key",
                    "cache_payload": {"dataset": "fixture"},
                    "samples": 2,
                    "bytes": tokens.len(),
                    "storage_schema_version": 6,
                    "byte_order": "little",
                    "token_dtype": "uint16",
                    "label_dtype": null,
                    "label_encoding": "input_ids_alias",
                    "label_ignore_sentinel": null,
                    "has_rosa_ids": true,
                    "rosa_dtype": "uint16",
                    "has_loss_weights": true,
                    "loss_weight_encoding": "float32_palette_rle",
                    "loss_weight_palette": [0.25, 1.0],
                    "ordered_record_sha256": ordered_sha,
                    "ordered_record_hash_algorithm": TOKEN_CACHE_RECORD_HASH_ALGORITHM,
                    "tokens_sha256": tokens_sha,
                    "portable_index_file": PORTABLE_TOKEN_CACHE_INDEX_FILENAME,
                    "portable_index_format": PORTABLE_TOKEN_CACHE_INDEX_FORMAT,
                    "portable_index_sha256": index_sha,
                    "audit_sha256": null
                }))?,
            )?;

            let loaded = load_training_dataset(&root)?;
            assert_eq!(loaded.rows.len(), 2);
            assert_eq!(loaded.rows[0].input_ids, vec![10, 11, 12]);
            assert_eq!(loaded.rows[0].labels, Some(vec![10, 11, 12]));
            assert_eq!(loaded.rows[0].loss_weights, Some(vec![0.25, 0.25, 1.0]));
            assert_eq!(loaded.rows[1].input_ids, vec![20, 21]);
            assert_eq!(loaded.rows[1].loss_weights, Some(vec![1.0, 1.0]));
            assert_eq!(loaded.identity.source_kind, "hierarchos-token-cache");
            assert_eq!(loaded.identity.total_tokens, 5);
            assert_eq!(
                loaded.identity.token_cache["ordered_record_sha256"],
                serde_json::Value::String(ordered_sha)
            );
            assert_eq!(
                loaded.identity.token_cache["tokens_sha256"],
                serde_json::Value::String(tokens_sha)
            );
            Ok(())
        })();
        let _ = fs::remove_dir_all(&root);
        result
    }

    #[test]
    fn exact_resume_identity_uses_full_content_for_legacy_self_digest_compatibility() -> Result<()>
    {
        let current = signed_test_run_identity(1.0e-4f32);
        let mut saved = current.clone();
        saved
            .as_object_mut()
            .expect("test run identity must be an object")
            .insert(
                "sha256".to_owned(),
                serde_json::Value::String("a".repeat(64)),
            );
        let saved: serde_json::Value = serde_json::from_slice(&serde_json::to_vec(&saved)?)?;
        validate_native_run_identity_value(&saved, &current)?;

        let changed = signed_test_run_identity(2.0e-4f32);
        let error = validate_native_run_identity_value(&saved, &changed)
            .expect_err("different exact-resume content must fail even with a well-shaped digest");
        assert!(error
            .to_string()
            .contains("exact native resume identity mismatch"));
        Ok(())
    }

    #[test]
    fn mid_epoch_replay_is_built_atomically_with_identity_and_running_carriers() -> Result<()> {
        let run_identity = signed_test_run_identity(1.0e-4f32);
        let scalar = |value| hierarchos_vulkan::HierarchosPortableReplayFloatTensor {
            shape: vec![1],
            values: vec![value],
        };
        let carriers = HierarchosPortableRunningCarriers {
            h_state: scalar(1.0),
            l_state: scalar(2.0),
            previous_context: scalar(3.0),
            target_context: scalar(4.0),
            drift_state: scalar(5.0),
            rosa_token_histories: vec![vec![7, 8]],
            ltm_state: HierarchosPortableLtmRunningState {
                fast_vals: None,
                mom_vals: None,
                timestamps: None,
                sources: None,
                wallclock_timestamps: None,
            },
        };
        let (encoded_state, tensors) =
            assemble_portable_replay_state(&run_identity, Some(&carriers))?;
        let session = HierarchosTrainingSessionState {
            format: HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT.to_string(),
            completed_epoch: 0,
            mid_epoch_step: 1,
            optimizer_grouping_version: 2,
            main_lr_scheduler: None,
            ltm_lr_scheduler: None,
            effective_training_config: serde_json::json!({"persist_state": true}),
            skipped_train_batches: 0,
            data_stream_cursor: Some(HierarchosDataStreamCursorState {
                format: HIERARCHOS_VULKAN_DATA_STREAM_CURSOR_FORMAT.to_string(),
                sampler_kind: "epoch-shuffle".to_string(),
                rng_algorithm: HIERARCHOS_VULKAN_PORTABLE_SAMPLER_RNG_ALGORITHM.to_string(),
                seed: 0,
                epoch: 0,
                batch_cursor: 1,
                dataset_size: 4,
                batch_size: 1,
                shuffle: false,
                drop_last: true,
                bucket_size: None,
                preserve_order: false,
            }),
            execution_policy: Some(HierarchosExecutionPolicyState {
                format: HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT.to_string(),
                source_backend: "vulkan".to_string(),
                compute_dtype: "float32".to_string(),
                autocast_enabled: false,
                stochastic_rng: HierarchosStochasticRngPolicyState {
                    mode: "none".to_string(),
                    state_required: false,
                    canonical_counter: None,
                },
                loss_scaling: HierarchosLossScalingState {
                    mode: "none".to_string(),
                    scale: None,
                    growth_factor: None,
                    backoff_factor: None,
                    growth_interval: None,
                    growth_tracker: None,
                    pending_gradients_scaled: false,
                },
            }),
        };

        let replay = HierarchosPortableTrainingReplay::new_with_training_session(
            0,
            1,
            session,
            encoded_state,
            tensors,
        )?;
        let keys = replay
            .encoded_state
            .get("items")
            .and_then(serde_json::Value::as_array)
            .context("test replay must encode dict items")?
            .iter()
            .filter_map(|pair| pair.as_array()?.first()?.as_str())
            .collect::<Vec<_>>();
        assert!(keys.contains(&"run_identity"));
        assert!(keys.contains(&"running_states"));
        assert!(!replay.tensors.is_empty());
        Ok(())
    }

    #[test]
    fn training_source_mode_is_explicit_and_mutually_exclusive() {
        let weights = PathBuf::from("weights-package");
        let checkpoint = PathBuf::from("resume-package");

        assert_eq!(
            resolve_training_source(Some(weights.clone()), None).unwrap(),
            (weights, false)
        );
        assert_eq!(
            resolve_training_source(None, Some(checkpoint.clone())).unwrap(),
            (checkpoint, true)
        );

        let error = resolve_training_source(
            Some(PathBuf::from("weights-package")),
            Some(PathBuf::from("resume-package")),
        )
        .expect_err("fresh training and exact resume must not be inferred simultaneously");
        assert!(error.to_string().contains("mutually exclusive"));

        let error = resolve_training_source(None, None)
            .expect_err("the trainer must require an explicit training source mode");
        assert!(error.to_string().contains("missing training source"));
    }

    #[test]
    fn exact_resume_preflight_rejects_weights_only_training_manifest() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hierarchos-vulkan-exact-resume-preflight-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join(HIERARCHOS_VULKAN_TRAINING_MANIFEST_FILENAME),
            serde_json::to_vec(&serde_json::json!({
                "format": "hierarchos-vulkan-training-state-v6",
                "architecture_revision": "coherent-v9",
                "model_file": "model.safetensors",
                "optimizer_file": "optimizer.safetensors",
                "optimizer_step": 0,
                "optimizer_tensor_count": 0,
                "training_step": 0,
                "training_precision_policy": "fp32"
            }))
            .unwrap(),
        )
        .unwrap();

        let error = validate_exact_resume_package(&root)
            .expect_err("weights-only package must not masquerade as exact continuation");
        assert!(error.to_string().contains("training_session"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn deferred_replica_broadcast_retirement_preserves_worker_telemetry() -> Result<()> {
        let (completion, completed) = mpsc::channel();
        let mut worker_summary = ReplicaStateBroadcastSummary::default();
        worker_summary.queue_submissions = 7;
        worker_summary.stream_chunks = 3;
        completion.send(ReplicaStateBroadcastCompletion {
            replica_index: 0,
            result: Ok(worker_summary),
        })?;
        drop(completion);

        let retirement = ReplicaStateBroadcastRetirement::Deferred(ReplicaStateBroadcastTicket {
            replica_count: 1,
            completed,
        });
        let summary = retirement.resolve()?;
        assert_eq!(summary.queue_submissions, 7);
        assert_eq!(summary.stream_chunks, 3);
        Ok(())
    }

    #[test]
    fn resume_numerical_policy_scalar_rejects_safety_drift() {
        let config = serde_json::json!({"grad_clip": 1.0});
        validate_resume_f32_config(&config, "grad_clip", 1.0, "--grad-clip")
            .expect("exactly matching safety scalar must resume");
        let error = validate_resume_f32_config(&config, "grad_clip", 0.5, "--grad-clip")
            .expect_err("changing gradient clipping across exact resume must fail closed");
        assert!(error
            .to_string()
            .contains("exact continuation forbids changing optimizer/objective safety policy"));

        validate_resume_f32_config(&serde_json::json!({}), "grad_clip", 1.0, "--grad-clip")
            .expect("legacy checkpoints without the new identity field remain loadable");
    }

    #[test]
    fn skipped_training_batch_budget_is_fail_closed_and_resume_counted() {
        let mut skipped = 0u64;
        account_skipped_training_batch(&mut skipped, 1, 2, 4, "synthetic non-finite window")
            .expect("first skipped window is inside the explicit budget");
        assert_eq!(skipped, 1);

        let error =
            account_skipped_training_batch(&mut skipped, 1, 2, 5, "synthetic non-finite window")
                .expect_err("second skipped window must exceed a budget of one");
        assert_eq!(skipped, 2);
        let message = error.to_string();
        assert!(message.contains("skip/error budget exceeded"));
        assert!(message.contains("Observed 2, allowed 1"));

        let mut resumed_over_budget = 3u64;
        let error = account_skipped_training_batch(
            &mut resumed_over_budget,
            3,
            7,
            8,
            "resumed synthetic non-finite window",
        )
        .expect_err("the first new skip after resuming at the budget must fail closed");
        assert_eq!(resumed_over_budget, 4);
        assert!(error.to_string().contains("Observed 4, allowed 3"));
    }

    fn row(tokens: &[u32]) -> DatasetRow {
        DatasetRow {
            input_ids: tokens.to_vec(),
            labels: Some(tokens.iter().map(|&token| i64::from(token)).collect()),
            attention_mask: None,
            loss_weights: None,
        }
    }

    fn row_with_len(tokens: usize) -> DatasetRow {
        row(&vec![1; tokens])
    }

    #[test]
    fn length_grouped_resume_uses_dataset_lengths_for_exact_mid_epoch_batch() -> Result<()> {
        let dataset = [5usize, 1, 4, 2, 6, 3, 9, 7, 8, 2, 5, 4]
            .into_iter()
            .map(row_with_len)
            .collect::<Vec<_>>();
        let lengths = dataset_row_lengths(&dataset)?;
        let mut cursor = HierarchosDataStreamCursorState {
            format: HIERARCHOS_VULKAN_DATA_STREAM_CURSOR_FORMAT.to_string(),
            sampler_kind: "length-grouped-batch".to_string(),
            rng_algorithm: HIERARCHOS_VULKAN_PORTABLE_SAMPLER_RNG_ALGORITHM.to_string(),
            seed: 123,
            epoch: 0,
            batch_cursor: 2,
            dataset_size: dataset.len() as u64,
            batch_size: 3,
            shuffle: true,
            drop_last: true,
            bucket_size: Some(6),
            preserve_order: false,
        };

        assert_eq!(
            cursor.current_batch_indices(Some(&lengths))?,
            Some(vec![0, 2, 9])
        );
        cursor.preserve_order = true;
        assert_eq!(
            cursor.current_batch_indices(Some(&lengths))?,
            Some(vec![10, 11, 9])
        );
        Ok(())
    }

    #[test]
    fn variable_length_batch_padding_is_masked_and_ignored() {
        let a = row(&[1, 2, 3]);
        let b = row(&[4, 5]);
        let packed = pack_batch(&[&a, &b], 8).unwrap();
        assert_eq!(packed.tokens, 3);
        assert_eq!(packed.input_ids, vec![1, 2, 3, 4, 5, 0]);
        assert_eq!(packed.labels, vec![1, 2, 3, 4, 5, -100]);
        assert_eq!(packed.attention_mask, vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(packed.loss_weights, vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn gradient_accumulation_closes_partial_final_group() {
        assert!(matches!(
            update_mode(0, 3, 2),
            HierarchosTokenTapeUpdateMode::BeginAccumulation
        ));
        assert!(matches!(
            update_mode(1, 3, 2),
            HierarchosTokenTapeUpdateMode::FinishAccumulation
        ));
        assert!(matches!(
            update_mode(2, 3, 2),
            HierarchosTokenTapeUpdateMode::Step
        ));
    }

    #[test]
    fn multi_device_ranges_are_contiguous_balanced_and_ordered() {
        assert_eq!(
            contiguous_weighted_shard_ranges(&[1; 7], 3).unwrap(),
            vec![0..2, 2..4, 4..7]
        );
        assert_eq!(
            contiguous_weighted_shard_ranges(&[1; 2], 4).unwrap(),
            vec![0..1, 1..2]
        );
    }

    #[test]
    fn multi_device_ranges_balance_variable_token_work_without_reordering() {
        assert_eq!(
            contiguous_weighted_shard_ranges(&[8, 1, 1, 1, 1, 1], 3).unwrap(),
            vec![0..1, 1..3, 3..6]
        );
        assert_eq!(
            contiguous_weighted_shard_ranges(&[1, 1, 7, 1, 1], 2).unwrap(),
            vec![0..2, 2..5]
        );
    }

    #[test]
    fn multi_device_ranges_follow_asymmetric_lane_capacity_without_reordering() {
        assert_eq!(
            contiguous_weighted_shard_ranges_by_capacity(&[1; 14], &[4.0, 2.0, 1.0]).unwrap(),
            vec![0..8, 8..12, 12..14]
        );
        assert_eq!(
            contiguous_weighted_shard_ranges_by_capacity(&[2, 2, 8, 2, 2], &[1.0, 3.0]).unwrap(),
            vec![0..2, 2..5]
        );
    }

    #[test]
    fn device_index_list_parser_preserves_declared_order() {
        assert_eq!(
            parse_device_indices(OsString::from("2, 0,5")).unwrap(),
            vec![2, 0, 5]
        );
        assert!(parse_device_indices(OsString::from("0,,1")).is_err());
    }

    #[test]
    fn lr_curve_matches_pytorch_warmup_cosine_geometry() {
        let total_steps = 4;
        let warmup_steps = 2;
        let samples = (0..=total_steps)
            .map(|step| scheduled_lr_at_step(1.0, 0.1, warmup_steps, total_steps, step))
            .collect::<Vec<_>>();
        let expected = [0.55, 1.0, 1.0, 0.55, 0.1];
        for (actual, expected) in samples.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn lr_scheduler_counter_advances_past_curve_horizon_at_terminal_lr() {
        let mut schedule = HierarchosLearningRateScheduleState {
            enabled: true,
            step: Some(2),
            total_steps: Some(2),
            max_lr: Some(1.0e-3),
            min_lr: Some(1.0e-5),
            warmup_steps: Some(0),
            warmup_ratio: Some(0.0),
            resolved_warmup_steps: Some(0),
            base_lrs: vec![1.0e-3, 1.0e-3],
            last_lrs: vec![1.0e-5, 1.0e-5],
            step_count: Some(3),
        };

        advance_lr_schedule(&mut schedule).unwrap();
        assert_eq!(schedule.step, Some(3));
        assert_eq!(schedule.step_count, Some(4));
        assert_eq!(schedule.last_lrs, vec![1.0e-5, 1.0e-5]);

        advance_lr_schedule(&mut schedule).unwrap();
        assert_eq!(schedule.step, Some(4));
        assert_eq!(schedule.step_count, Some(5));
        assert_eq!(schedule.last_lrs, vec![1.0e-5, 1.0e-5]);
    }

    #[test]
    fn warmup_ratio_uses_pytorch_ceil_and_epoch_cap() {
        assert_eq!(resolve_warmup_steps(0, 0.21, 10), 3);
        assert_eq!(resolve_warmup_steps(99, 0.0, 10), 9);
        assert_eq!(resolve_warmup_steps(0, 1.0, 1), 0);
    }

    #[test]
    fn joint_runtime_candidates_only_reduce_preflight_memory_geometry() {
        let baseline = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 8,
            state_checkpoint_stride: 2,
        };
        let arms = JointRuntimeAutotuner::candidate_arms(1024, baseline, 16);
        assert!(!arms.is_empty());
        assert!(arms.iter().all(|arm| {
            arm.gradient_stream_chunk_values <= 1024
                && arm.tape_geometry.sequence_microbatch_size <= 8
                && arm.tape_geometry.state_checkpoint_stride >= 2
                && arm.tape_geometry.state_checkpoint_stride <= 16
        }));
        assert!(arms.iter().any(|arm| arm.optimizer_broadcast_overlap));
        assert!(arms.iter().any(|arm| !arm.optimizer_broadcast_overlap));
        for arm in &arms {
            assert!(arms.iter().any(|peer| {
                peer.gradient_stream_chunk_values == arm.gradient_stream_chunk_values
                    && peer.tape_geometry == arm.tape_geometry
                    && peer.optimizer_broadcast_overlap != arm.optimizer_broadcast_overlap
            }));
        }
    }

    #[test]
    fn joint_runtime_arm_switch_dwells_before_scoring() {
        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 4,
            state_checkpoint_stride: 1,
        };
        let arms = vec![
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 1024,
                tape_geometry,
                optimizer_broadcast_overlap: true,
            },
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 512,
                tape_geometry,
                optimizer_broadcast_overlap: true,
            },
        ];
        let mut tuner = JointRuntimeAutotuner {
            measurements: vec![JointRuntimeArmMeasurements::default(); arms.len()],
            current_run_selected_windows: vec![0; arms.len()],
            current_run_scored_windows: vec![0; arms.len()],
            arms,
            explore_every: 16,
            selection_step: 0,
            last_selected_index: None,
            forced_followup_index: None,
            warmup_windows_remaining: 0,
            initial_preferred_index: None,
            locked_index: None,
        };

        let (first_index, first_arm, first_scores) = tuner.select();
        assert_eq!(first_index, 0);
        assert!(!first_scores);
        let (followup_index, followup_arm, followup_scores) = tuner.select();
        assert_eq!(followup_index, first_index);
        assert_eq!(followup_arm, first_arm);
        assert!(!followup_scores);
        let (steady_index, steady_arm, steady_scores) = tuner.select();
        assert_eq!(steady_index, first_index);
        assert_eq!(steady_arm, first_arm);
        assert!(steady_scores);
        tuner.observe_with_phase(
            steady_index,
            1_000,
            1.0,
            10,
            SubmissionLatencyDelta {
                samples: 2,
                total_ns: 200,
                ..SubmissionLatencyDelta::default()
            },
            &[],
            JointRuntimePhaseWindowProfile {
                gradient_reduction_service_seconds: 0.020,
                optimizer_boundary_service_seconds: 0.010,
            },
        );

        let (confidence_repeat_index, confidence_repeat_arm, confidence_repeat_scores) =
            tuner.select();
        assert_eq!(confidence_repeat_index, first_index);
        assert_eq!(confidence_repeat_arm, first_arm);
        assert!(confidence_repeat_scores);
        tuner.observe_with_phase(
            confidence_repeat_index,
            1_000,
            1.0,
            10,
            SubmissionLatencyDelta::default(),
            &[],
            JointRuntimePhaseWindowProfile {
                gradient_reduction_service_seconds: 0.020,
                optimizer_boundary_service_seconds: 0.010,
            },
        );

        let (second_index, _, second_scores) = tuner.select();
        assert_eq!(second_index, 1);
        assert!(!second_scores);
        let (second_followup_index, _, second_followup_scores) = tuner.select();
        assert_eq!(second_followup_index, second_index);
        assert!(!second_followup_scores);
        let (second_steady_index, _, second_steady_scores) = tuner.select();
        assert_eq!(second_steady_index, second_index);
        assert!(second_steady_scores);
    }

    #[test]
    fn joint_runtime_device_memory_pressure_telemetry_tracks_peak_and_high_pressure() {
        let mut measurements = JointRuntimeDeviceMeasurements::default();
        measurements.observe(&JointRuntimeDeviceWindowProfile {
            lane_index: 0,
            tokens: 1_000,
            elapsed_seconds: 1.0,
            device_local_usage_ratio: Some(0.80),
            device_local_pressure_bucket: Some(6),
            ..JointRuntimeDeviceWindowProfile::default()
        });

        assert_eq!(measurements.adaptive_device_local_usage_ratio, Some(0.80));
        assert_eq!(measurements.peak_device_local_usage_ratio, Some(0.80));
        assert_eq!(measurements.max_device_local_pressure_bucket, Some(6));
        assert_eq!(measurements.high_memory_pressure_windows, 1);
        let report = measurements.report();
        assert_eq!(report["max_device_local_pressure_bucket"], 6);
        assert_eq!(report["high_memory_pressure_windows"], 1);
    }

    #[test]
    fn joint_runtime_throttle_detector_requires_throughput_drop_and_gpu_cost_rise() {
        let mut measurements = JointRuntimeArmMeasurements::default();
        let stable_latency = SubmissionLatencyDelta {
            kernel_profile_samples: 1,
            kernel_dispatches: 1,
            kernel_gpu_ns_total: 1_000_000,
            ..SubmissionLatencyDelta::default()
        };
        for _ in 0..2 {
            measurements.observe_with_phase(
                1_000,
                1.0,
                1,
                stable_latency,
                &[],
                JointRuntimePhaseWindowProfile::default(),
            );
        }
        assert!(!measurements.last_window_throttle_suspect);

        measurements.observe_with_phase(
            1_000,
            1.5,
            1,
            SubmissionLatencyDelta {
                kernel_profile_samples: 1,
                kernel_dispatches: 1,
                kernel_gpu_ns_total: 1_300_000,
                ..SubmissionLatencyDelta::default()
            },
            &[],
            JointRuntimePhaseWindowProfile::default(),
        );
        assert!(measurements.last_window_throttle_suspect);
        assert_eq!(measurements.throttle_suspect_windows, 1);

        measurements.observe_with_phase(
            1_000,
            1.5,
            1,
            stable_latency,
            &[],
            JointRuntimePhaseWindowProfile::default(),
        );
        assert!(!measurements.last_window_throttle_suspect);
        assert_eq!(measurements.throttle_suspect_windows, 1);
    }

    #[test]
    fn joint_runtime_heterogeneous_lane_efficiency_penalizes_slow_replica() {
        let mut measurements = JointRuntimeArmMeasurements::default();
        let device_windows = vec![
            JointRuntimeDeviceWindowProfile {
                lane_index: 0,
                tokens: 500,
                elapsed_seconds: 0.4,
                ..JointRuntimeDeviceWindowProfile::default()
            },
            JointRuntimeDeviceWindowProfile {
                lane_index: 1,
                tokens: 500,
                elapsed_seconds: 0.8,
                ..JointRuntimeDeviceWindowProfile::default()
            },
        ];
        measurements.observe(
            1_000,
            1.0,
            10,
            SubmissionLatencyDelta::default(),
            &device_windows,
        );
        assert_eq!(measurements.adaptive_tokens_per_second, Some(1_000.0));
        assert!(
            (measurements
                .adaptive_effective_tokens_per_second
                .expect("effective throughput should be observed")
                - 750.0)
                .abs()
                < 1.0e-9
        );
        assert_eq!(measurements.devices.len(), 2);
        assert_eq!(
            measurements.devices[1].adaptive_tokens_per_second,
            Some(625.0)
        );
    }

    #[test]
    fn joint_runtime_device_share_learning_prefers_faster_lane_after_confidence_ramp() {
        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 4,
            state_checkpoint_stride: 1,
        };
        let arms = vec![JointRuntimeScheduleArm {
            gradient_stream_chunk_values: 1024,
            tape_geometry,
            optimizer_broadcast_overlap: true,
        }];
        let mut measurements = JointRuntimeArmMeasurements::default();
        for _ in 0..4 {
            measurements.observe(
                1_000,
                0.4,
                10,
                SubmissionLatencyDelta::default(),
                &[
                    JointRuntimeDeviceWindowProfile {
                        lane_index: 0,
                        tokens: 800,
                        elapsed_seconds: 0.4,
                        ..JointRuntimeDeviceWindowProfile::default()
                    },
                    JointRuntimeDeviceWindowProfile {
                        lane_index: 1,
                        tokens: 200,
                        elapsed_seconds: 0.4,
                        ..JointRuntimeDeviceWindowProfile::default()
                    },
                ],
            );
        }
        let tuner = JointRuntimeAutotuner {
            measurements: vec![measurements],
            current_run_selected_windows: vec![0; arms.len()],
            current_run_scored_windows: vec![0; arms.len()],
            arms,
            explore_every: 16,
            selection_step: 0,
            last_selected_index: None,
            forced_followup_index: None,
            warmup_windows_remaining: 0,
            initial_preferred_index: None,
            locked_index: None,
        };
        let workload_weights = tuner.lane_workload_weights(0, 2);
        assert_eq!(workload_weights.len(), 2);
        assert!(workload_weights[0] > workload_weights[1] * 3.5);
        assert!(workload_weights[0] > 1.0);
        assert!(workload_weights[1] < 1.0);
    }

    #[test]
    fn joint_runtime_device_share_learning_penalizes_lane_variance() {
        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 4,
            state_checkpoint_stride: 1,
        };
        let arms = vec![JointRuntimeScheduleArm {
            gradient_stream_chunk_values: 1024,
            tape_geometry,
            optimizer_broadcast_overlap: true,
        }];
        let measurements = JointRuntimeArmMeasurements {
            devices: vec![
                JointRuntimeDeviceMeasurements {
                    lane_index: 0,
                    windows: 4,
                    adaptive_tokens_per_second: Some(1_000.0),
                    confidence_throughput_samples: 4,
                    confidence_throughput_mean: Some(1_000.0),
                    confidence_throughput_m2: 0.0,
                    ..JointRuntimeDeviceMeasurements::default()
                },
                JointRuntimeDeviceMeasurements {
                    lane_index: 1,
                    windows: 4,
                    adaptive_tokens_per_second: Some(1_000.0),
                    confidence_throughput_samples: 4,
                    confidence_throughput_mean: Some(1_000.0),
                    // Sample variance = 1,000,000 tokens^2/s^2: the mean is
                    // identical to lane 0, but the lower confidence bound is
                    // intentionally much worse.
                    confidence_throughput_m2: 3_000_000.0,
                    ..JointRuntimeDeviceMeasurements::default()
                },
            ],
            ..JointRuntimeArmMeasurements::default()
        };
        let stable = measurements.devices[0]
            .confidence_adjusted_tokens_per_second()
            .expect("stable lane should have a confidence estimate");
        let noisy = measurements.devices[1]
            .confidence_adjusted_tokens_per_second()
            .expect("noisy lane should have a confidence estimate");
        assert!(stable > noisy * 4.0);

        let tuner = JointRuntimeAutotuner {
            measurements: vec![measurements],
            current_run_selected_windows: vec![0; arms.len()],
            current_run_scored_windows: vec![0; arms.len()],
            arms,
            explore_every: 16,
            selection_step: 0,
            last_selected_index: None,
            forced_followup_index: None,
            warmup_windows_remaining: 0,
            initial_preferred_index: None,
            locked_index: None,
        };
        let workload_weights = tuner.lane_workload_weights(0, 2);
        assert!(workload_weights[0] > 1.0);
        assert!(workload_weights[1] < 1.0);
        assert!(workload_weights[0] > workload_weights[1] * 4.0);
    }

    #[test]
    fn joint_runtime_phase_service_signals_adapt_independently() {
        let mut measurements = JointRuntimeArmMeasurements::default();
        measurements.observe_with_phase(
            1_000,
            1.0,
            10,
            SubmissionLatencyDelta::default(),
            &[],
            JointRuntimePhaseWindowProfile {
                gradient_reduction_service_seconds: 0.020,
                optimizer_boundary_service_seconds: 0.030,
            },
        );
        measurements.observe_with_phase(
            1_000,
            1.0,
            10,
            SubmissionLatencyDelta::default(),
            &[],
            JointRuntimePhaseWindowProfile {
                gradient_reduction_service_seconds: 0.010,
                optimizer_boundary_service_seconds: 0.020,
            },
        );
        assert!(
            (measurements
                .adaptive_gradient_reduction_service_seconds
                .unwrap()
                - 0.0175)
                .abs()
                < 1.0e-12
        );
        assert!(
            (measurements
                .adaptive_optimizer_boundary_service_seconds
                .unwrap()
                - 0.0275)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn joint_runtime_phase_selectors_compose_winners_from_different_arms() {
        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 4,
            state_checkpoint_stride: 1,
        };
        let arms = vec![
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 1024,
                tape_geometry,
                optimizer_broadcast_overlap: true,
            },
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 512,
                tape_geometry,
                optimizer_broadcast_overlap: false,
            },
        ];
        let mut reduction_winner = JointRuntimeArmMeasurements::default();
        let mut optimizer_winner = JointRuntimeArmMeasurements::default();
        for _ in 0..JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS {
            reduction_winner.observe_with_phase(
                1_000,
                1.0,
                10,
                SubmissionLatencyDelta::default(),
                &[],
                JointRuntimePhaseWindowProfile {
                    gradient_reduction_service_seconds: 0.010,
                    optimizer_boundary_service_seconds: 0.040,
                },
            );
            optimizer_winner.observe_with_phase(
                1_000,
                1.0,
                10,
                SubmissionLatencyDelta::default(),
                &[],
                JointRuntimePhaseWindowProfile {
                    gradient_reduction_service_seconds: 0.020,
                    optimizer_boundary_service_seconds: 0.010,
                },
            );
        }
        let tuner = JointRuntimeAutotuner {
            measurements: vec![reduction_winner, optimizer_winner],
            current_run_selected_windows: vec![0; arms.len()],
            current_run_scored_windows: vec![0; arms.len()],
            arms,
            explore_every: 16,
            selection_step: 0,
            last_selected_index: None,
            forced_followup_index: None,
            warmup_windows_remaining: 0,
            initial_preferred_index: None,
            locked_index: None,
        };
        assert_eq!(tuner.best_gradient_transport_chunk_values(), Some(1024));
        assert_eq!(tuner.best_optimizer_broadcast_overlap(), Some(false));
        let persisted = tuner
            .persisted_profile(JointRuntimeProfileKey {
                architecture_revision: "factorized-test".to_owned(),
                architecture_contract_sha256: None,
                batch_size: 1,
                gradient_accumulation_steps: 1,
                tokens_per_sequence: 8,
                device_uuids: vec!["gpu-a".to_owned(), "gpu-b".to_owned()],
                driver_uuids: vec!["driver-a".to_owned(), "driver-b".to_owned()],
                transport_backends: vec!["device-group-peer-memory-v1".to_owned()],
            })
            .expect("factorized evidence should produce a persisted winner");
        assert_eq!(persisted.winning_arm.gradient_stream_chunk_values, 1024);
        assert!(!persisted.winning_arm.optimizer_broadcast_overlap);
        assert!(
            !tuner.arms.contains(&persisted.winning_arm),
            "the persisted winner should prove the selectors can synthesize a non-Cartesian combination"
        );
    }

    #[test]
    fn joint_runtime_tape_selectors_compose_unmeasured_microbatch_stride_pair() {
        let arms = vec![
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 1024,
                tape_geometry: PhaseAwareTapeGeometry {
                    sequence_microbatch_size: 4,
                    state_checkpoint_stride: 1,
                },
                optimizer_broadcast_overlap: true,
            },
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 1024,
                tape_geometry: PhaseAwareTapeGeometry {
                    sequence_microbatch_size: 2,
                    state_checkpoint_stride: 1,
                },
                optimizer_broadcast_overlap: true,
            },
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 1024,
                tape_geometry: PhaseAwareTapeGeometry {
                    sequence_microbatch_size: 4,
                    state_checkpoint_stride: 2,
                },
                optimizer_broadcast_overlap: true,
            },
        ];
        let mut measurements = Vec::new();
        for tokens_per_second in [50.0, 300.0, 250.0] {
            let mut arm = JointRuntimeArmMeasurements::default();
            for _ in 0..JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS {
                arm.observe(
                    tokens_per_second as u64,
                    1.0,
                    10,
                    SubmissionLatencyDelta::default(),
                    &[],
                );
            }
            measurements.push(arm);
        }
        let tuner = JointRuntimeAutotuner {
            measurements,
            current_run_selected_windows: vec![0; arms.len()],
            current_run_scored_windows: vec![0; arms.len()],
            arms,
            explore_every: 16,
            selection_step: 0,
            last_selected_index: None,
            forced_followup_index: None,
            warmup_windows_remaining: 0,
            initial_preferred_index: None,
            locked_index: None,
        };

        let geometry = tuner
            .best_tape_geometry()
            .expect("factorized tape evidence should synthesize a geometry");
        assert_eq!(geometry.sequence_microbatch_size, 2);
        assert_eq!(geometry.state_checkpoint_stride, 2);
        assert!(
            !tuner.arms.iter().any(|arm| arm.tape_geometry == geometry),
            "microbatch and checkpoint stride should compose without measuring their full Cartesian pair"
        );
    }

    #[test]
    fn joint_runtime_phase_selector_priority_is_unitless_and_bootstrap_aware() {
        let gradient = [JointRuntimeFactorizedScore {
            value: 1024usize,
            observed_windows: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS,
            adaptive_score: Some(100.0),
            confidence_adjusted_score: Some(90.0),
            exploration_score: Some(99.0),
            relative_uncertainty: Some(0.01),
            effective_samples: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS as f64,
            observations_since_last_measurement: 1,
        }];
        let optimizer = [JointRuntimeFactorizedScore {
            value: true,
            observed_windows: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS,
            adaptive_score: Some(10.0),
            confidence_adjusted_score: Some(8.0),
            exploration_score: Some(14.0),
            relative_uncertainty: Some(0.10),
            effective_samples: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS as f64,
            observations_since_last_measurement: 5,
        }];
        let tape_microbatch = [JointRuntimeFactorizedScore {
            value: 4usize,
            observed_windows: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS,
            adaptive_score: Some(10_000.0),
            confidence_adjusted_score: Some(9_000.0),
            exploration_score: Some(10_800.0),
            relative_uncertainty: Some(0.02),
            effective_samples: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS as f64,
            observations_since_last_measurement: 2,
        }];
        let mut tape_stride = [JointRuntimeFactorizedScore {
            value: 1usize,
            observed_windows: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS,
            adaptive_score: Some(10_000.0),
            confidence_adjusted_score: Some(9_500.0),
            exploration_score: Some(10_400.0),
            relative_uncertainty: Some(0.01),
            effective_samples: JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS as f64,
            observations_since_last_measurement: 1,
        }];

        assert_eq!(
            JointRuntimeAutotuner::active_factorized_selector(
                9,
                true,
                &gradient,
                &optimizer,
                &tape_microbatch,
                &tape_stride,
            ),
            Some(1),
            "raw service-rate/tokens-per-second magnitude must not dominate normalized UCB pressure"
        );

        tape_stride[0].observed_windows = JOINT_RUNTIME_MIN_CONFIDENCE_WINDOWS - 1;
        assert_eq!(
            JointRuntimeAutotuner::active_factorized_selector(
                10,
                false,
                &gradient,
                &optimizer,
                &tape_microbatch,
                &tape_stride,
            ),
            Some(3),
            "bootstrap debt must preempt exploitation until every phase selector has enough evidence"
        );
    }

    #[test]
    fn joint_runtime_confidence_ranking_prefers_repeatable_arm_over_lucky_one_off() {
        let mut lucky = JointRuntimeArmMeasurements::default();
        lucky.observe(1_200, 1.0, 10, SubmissionLatencyDelta::default(), &[]);
        let mut repeatable = JointRuntimeArmMeasurements::default();
        for _ in 0..2 {
            repeatable.observe(1_100, 1.0, 10, SubmissionLatencyDelta::default(), &[]);
        }

        assert_eq!(lucky.adaptive_effective_tokens_per_second, Some(1_200.0));
        assert_eq!(
            repeatable.adaptive_effective_tokens_per_second,
            Some(1_100.0)
        );
        assert!(
            repeatable
                .confidence_adjusted_effective_tokens_per_second_at(3)
                .unwrap()
                > lucky
                    .confidence_adjusted_effective_tokens_per_second_at(3)
                    .unwrap()
        );
        assert_eq!(
            JointRuntimeAutotuner::compare_measurements(&repeatable, &lucky, 3),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn joint_runtime_decayed_ucb_reopens_stale_winner() {
        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 4,
            state_checkpoint_stride: 1,
        };
        let arms = vec![
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 1024,
                tape_geometry,
                optimizer_broadcast_overlap: true,
            },
            JointRuntimeScheduleArm {
                gradient_stream_chunk_values: 512,
                tape_geometry,
                optimizer_broadcast_overlap: true,
            },
        ];
        let mut stale_winner = JointRuntimeArmMeasurements::default();
        for _ in 0..2 {
            stale_winner.observe(1_200, 1.0, 10, SubmissionLatencyDelta::default(), &[]);
        }
        stale_winner.last_observation_ordinal = 2;

        let mut recent_repeatable = JointRuntimeArmMeasurements::default();
        for _ in 0..10 {
            recent_repeatable.observe(1_100, 1.0, 10, SubmissionLatencyDelta::default(), &[]);
        }
        recent_repeatable.last_observation_ordinal = 12;

        let tuner = JointRuntimeAutotuner {
            measurements: vec![stale_winner, recent_repeatable],
            current_run_selected_windows: vec![0; arms.len()],
            current_run_scored_windows: vec![0; arms.len()],
            arms,
            explore_every: 16,
            selection_step: 16,
            last_selected_index: None,
            forced_followup_index: None,
            warmup_windows_remaining: 0,
            initial_preferred_index: None,
            locked_index: None,
        };
        let total_observations = tuner.total_scored_observations();
        assert_eq!(total_observations, 12);
        assert_eq!(tuner.best_measured_index(), Some(1));
        assert_eq!(tuner.best_exploration_index(), Some(0));
        assert!(
            tuner.measurements[0].effective_confidence_samples(total_observations)
                < tuner.measurements[1].effective_confidence_samples(total_observations)
        );
        assert!(
            tuner.measurements[0]
                .exploration_score_tokens_per_second_at(total_observations)
                .unwrap()
                > tuner.measurements[1]
                    .exploration_score_tokens_per_second_at(total_observations)
                    .unwrap()
        );
    }

    #[test]
    fn joint_runtime_persisted_winner_is_preferred_on_restart() {
        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 4,
            state_checkpoint_stride: 2,
        };
        let winner = JointRuntimeScheduleArm {
            gradient_stream_chunk_values: 512,
            tape_geometry,
            optimizer_broadcast_overlap: false,
        };
        let profile = PersistedJointRuntimeProfile {
            schema_version: JOINT_RUNTIME_PROFILE_SCHEMA_VERSION,
            profile_key: JointRuntimeProfileKey {
                architecture_revision: "test".to_owned(),
                architecture_contract_sha256: None,
                batch_size: 1,
                gradient_accumulation_steps: 1,
                tokens_per_sequence: 8,
                device_uuids: vec!["a".to_owned(), "b".to_owned()],
                driver_uuids: vec!["d".to_owned(), "d".to_owned()],
                transport_backends: vec!["device-group-peer-memory-v1".to_owned()],
            },
            winning_arm: winner,
            arms: vec![winner],
            measurements: vec![JointRuntimeArmMeasurements {
                windows: 3,
                tokens: 3_000,
                elapsed_seconds: 3.0,
                adaptive_tokens_per_second: Some(1_000.0),
                adaptive_effective_tokens_per_second: Some(950.0),
                ..JointRuntimeArmMeasurements::default()
            }],
        };
        let mut tuner = JointRuntimeAutotuner::new(1024, tape_geometry, 8, Some(&profile), false)
            .expect("joint runtime autotuner should be enabled in tests");
        let (_, selected, scored) = tuner.select();
        assert_eq!(selected, winner);
        assert!(!scored);
        let (_, selected_followup, scored_followup) = tuner.select();
        assert_eq!(selected_followup, winner);
        assert!(!scored_followup);
        let (_, selected_steady, scored_steady) = tuner.select();
        assert_eq!(selected_steady, winner);
        assert!(scored_steady);
    }

    #[test]
    fn joint_runtime_locked_profile_replays_winner_without_exploration() {
        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 4,
            state_checkpoint_stride: 2,
        };
        let winner = JointRuntimeScheduleArm {
            gradient_stream_chunk_values: 512,
            tape_geometry,
            optimizer_broadcast_overlap: false,
        };
        let profile = PersistedJointRuntimeProfile {
            schema_version: JOINT_RUNTIME_PROFILE_SCHEMA_VERSION,
            profile_key: JointRuntimeProfileKey {
                architecture_revision: "locked-test".to_owned(),
                architecture_contract_sha256: None,
                batch_size: 1,
                gradient_accumulation_steps: 1,
                tokens_per_sequence: 8,
                device_uuids: vec!["a".to_owned(), "b".to_owned()],
                driver_uuids: vec!["d".to_owned(), "d".to_owned()],
                transport_backends: vec!["device-group-peer-memory-v1".to_owned()],
            },
            winning_arm: winner,
            arms: vec![winner],
            measurements: vec![JointRuntimeArmMeasurements {
                windows: 3,
                tokens: 3_000,
                elapsed_seconds: 3.0,
                adaptive_tokens_per_second: Some(1_000.0),
                adaptive_effective_tokens_per_second: Some(950.0),
                ..JointRuntimeArmMeasurements::default()
            }],
        };
        let mut tuner = JointRuntimeAutotuner::new(1024, tape_geometry, 8, Some(&profile), true)
            .expect("locked joint runtime profile should initialize");
        assert_eq!(tuner.locked_arm(), Some(winner));
        for step in 0..64 {
            let (index, selected, scored) = tuner.select();
            assert_eq!(selected, winner);
            assert_eq!(
                scored,
                step >= usize::from(JOINT_RUNTIME_WARMUP_WINDOWS_AFTER_SWITCH)
            );
            if scored {
                tuner.observe_with_phase(
                    index,
                    1_000,
                    1.0,
                    10,
                    SubmissionLatencyDelta::default(),
                    &[],
                    JointRuntimePhaseWindowProfile::default(),
                );
            }
        }
        let report = tuner.report();
        let expected_scored = 64u64 - u64::from(JOINT_RUNTIME_WARMUP_WINDOWS_AFTER_SWITCH);
        assert_eq!(report["current_run_selected_windows"], 64);
        assert_eq!(report["current_run_scored_windows"], expected_scored);
        assert_eq!(report["arms"][0]["current_run_selected_windows"], 64);
        assert_eq!(
            report["arms"][0]["current_run_scored_windows"],
            expected_scored
        );
        let persisted = tuner
            .persisted_profile(profile.profile_key.clone())
            .expect("locked profile should remain persistable");
        assert_eq!(persisted.winning_arm, winner);
    }

    #[test]
    fn joint_runtime_profile_staged_writer_replaces_existing_snapshot() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hierarchos-vulkan-joint-runtime-profile-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();

        let tape_geometry = PhaseAwareTapeGeometry {
            sequence_microbatch_size: 2,
            state_checkpoint_stride: 1,
        };
        let arm = JointRuntimeScheduleArm {
            gradient_stream_chunk_values: 256,
            tape_geometry,
            optimizer_broadcast_overlap: true,
        };
        let profile_key = JointRuntimeProfileKey {
            architecture_revision: "durability-test".to_owned(),
            architecture_contract_sha256: Some("abc".to_owned()),
            batch_size: 2,
            gradient_accumulation_steps: 1,
            tokens_per_sequence: 4,
            device_uuids: vec!["gpu-a".to_owned(), "gpu-b".to_owned()],
            driver_uuids: vec!["driver-a".to_owned(), "driver-b".to_owned()],
            transport_backends: vec!["host-visible-staged-v2-pipelined".to_owned()],
        };
        let mut profile = PersistedJointRuntimeProfile {
            schema_version: JOINT_RUNTIME_PROFILE_SCHEMA_VERSION,
            profile_key: profile_key.clone(),
            winning_arm: arm,
            arms: vec![arm],
            measurements: vec![JointRuntimeArmMeasurements {
                windows: 1,
                tokens: 1_000,
                elapsed_seconds: 1.0,
                adaptive_tokens_per_second: Some(1_000.0),
                adaptive_effective_tokens_per_second: Some(900.0),
                ..JointRuntimeArmMeasurements::default()
            }],
        };

        let first_path = write_joint_runtime_profile(&root, &profile).unwrap();
        assert!(first_path.is_file());
        profile.measurements[0].windows = 9;
        profile.measurements[0].tokens = 9_000;
        let second_path = write_joint_runtime_profile(&root, &profile).unwrap();
        assert_eq!(first_path, second_path);

        let loaded = load_joint_runtime_profile(&root, &profile_key)
            .unwrap()
            .expect("rewritten joint runtime profile should load");
        assert_eq!(loaded.measurements[0].windows, 9);
        assert_eq!(loaded.measurements[0].tokens, 9_000);
        let direct_loaded = load_joint_runtime_profile_path(&second_path, &profile_key)
            .unwrap()
            .expect("explicit joint runtime profile path should load");
        assert_eq!(direct_loaded.measurements[0].windows, 9);
        let mut mismatched_key = profile_key.clone();
        mismatched_key.batch_size += 1;
        assert!(
            load_joint_runtime_profile_path(&second_path, &mismatched_key)
                .unwrap()
                .is_none()
        );
        assert!(!root
            .join(format!(
                ".{JOINT_RUNTIME_PROFILE_FILENAME}.tmp-{}",
                std::process::id()
            ))
            .exists());

        fs::remove_dir_all(root).unwrap();
    }
}
