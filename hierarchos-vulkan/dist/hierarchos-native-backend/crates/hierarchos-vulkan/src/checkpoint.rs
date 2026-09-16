use std::{collections::HashMap, fs, path::Path};

use anyhow::{bail, Context, Result};
use hierarchos_inference::{RosaState, RosaStateSnapshot, RosaTransitionSnapshot};
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, SafeTensors, TensorView},
};

use crate::rwkv_optimizer::{
    AdamWDecayClass, AdamWOptimizerSlotState, AdamWOptimizerState, RwkvParameterSnapshot,
};

/// Tensor payload stored beside the JSON host-replay document. U8 covers the
/// byte-generator states used by PyTorch CPU/CUDA RNGs; F32 covers recurrent
/// carrier tensors without forcing a model-layout conversion.
#[derive(Clone, Debug, PartialEq)]
pub enum HierarchosPortableReplayTensorData {
    U8(Vec<u8>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosPortableReplayTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: HierarchosPortableReplayTensorData,
}

/// One floating tensor decoded from the portable replay SafeTensors sidecar.
///
/// Replay tensors are promoted to FP32 on load so a checkpoint written by a
/// PyTorch FP16/BF16 autocast run enters the same canonical host boundary as a
/// native Vulkan checkpoint. `shape` retains the PyTorch row-major geometry so
/// callers can reject a same-length but semantically incompatible carrier.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosPortableReplayFloatTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosPortableReplayF64Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosPortableReplayI64Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<i64>,
}

/// The complete seven-part PyTorch coherent-v9 LTM continuation carrier.
///
/// Vulkan currently treats these values as read-only training-time state, but
/// they still belong to the exact-resume ABI. Keeping them in the portable
/// carrier prevents a PyTorch -> Vulkan -> PyTorch handoff from silently
/// discarding Titans momentum or filtering metadata that the native kernels do
/// not otherwise need to materialize.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosPortableLtmRunningState {
    pub fast_vals: Option<HierarchosPortableReplayFloatTensor>,
    pub mom_vals: Option<HierarchosPortableReplayFloatTensor>,
    pub timestamps: Option<HierarchosPortableReplayFloatTensor>,
    pub sources: Option<HierarchosPortableReplayI64Tensor>,
    pub wallclock_timestamps: Option<HierarchosPortableReplayF64Tensor>,
}

impl HierarchosPortableLtmRunningState {
    /// Validate the exact shared coherent-v9 LTM geometry consumed by the
    /// native Vulkan frontend. PyTorch can represent batch-isolated writable
    /// LTM state too, but that is a different execution policy and must not be
    /// silently flattened into the shared native cache during a backend handoff.
    pub fn validate_exact_shared_geometry(&self, slots: usize, value_dim: usize) -> Result<()> {
        let matrix_shape = vec![slots, value_dim];
        let vector_shape = vec![slots];
        let fast_vals = self
            .fast_vals
            .as_ref()
            .context("exact persisted LTM state is missing fast_vals")?;
        let mom_vals = self
            .mom_vals
            .as_ref()
            .context("exact persisted LTM state is missing _mom_vals")?;
        validate_replay_float_tensor("ltm.fast_vals", fast_vals)?;
        validate_replay_float_tensor("ltm._mom_vals", mom_vals)?;
        if fast_vals.shape != matrix_shape {
            bail!(
                "portable ltm.fast_vals shape {:?} does not match native shared LTM geometry {:?}",
                fast_vals.shape,
                matrix_shape
            );
        }
        if mom_vals.shape != matrix_shape {
            bail!(
                "portable ltm._mom_vals shape {:?} does not match native shared LTM geometry {:?}",
                mom_vals.shape,
                matrix_shape
            );
        }
        if let Some(timestamps) = self.timestamps.as_ref() {
            validate_replay_float_tensor("ltm.timestamps", timestamps)?;
            if timestamps.shape != vector_shape {
                bail!(
                    "portable ltm.timestamps shape {:?} does not match native shared LTM geometry {:?}",
                    timestamps.shape,
                    vector_shape
                );
            }
        }
        if let Some(sources) = self.sources.as_ref() {
            validate_replay_i64_tensor("ltm.sources", sources)?;
            if sources.shape != vector_shape {
                bail!(
                    "portable ltm.sources shape {:?} does not match native shared LTM geometry {:?}",
                    sources.shape,
                    vector_shape
                );
            }
        }
        if let Some(wallclock) = self.wallclock_timestamps.as_ref() {
            validate_replay_f64_tensor("ltm.wallclock_timestamps", wallclock)?;
            if wallclock.shape != vector_shape {
                bail!(
                    "portable ltm.wallclock_timestamps shape {:?} does not match native shared LTM geometry {:?}",
                    wallclock.shape,
                    vector_shape
                );
            }
        }
        Ok(())
    }
}

/// Backend-neutral recurrent state needed to continue a persistent Hierarchos
/// stream in the middle of an epoch.
///
/// PyTorch stores this inside the six-part `running_states` tuple. Native
/// Vulkan owns the H/L/context values at the trainer boundary and reconstructs
/// its device-resident bounded ROSA state from the validated per-lane token
/// histories. Drift and the full LTM continuation metadata remain part of the
/// portable carrier so a Vulkan -> PyTorch/CUDA -> Vulkan handoff is lossless.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosPortableRunningCarriers {
    pub h_state: HierarchosPortableReplayFloatTensor,
    pub l_state: HierarchosPortableReplayFloatTensor,
    pub previous_context: HierarchosPortableReplayFloatTensor,
    pub target_context: HierarchosPortableReplayFloatTensor,
    /// Python persists this value even for coherent-v9. The Vulkan training
    /// graph derives the token-zero drift seed from the restored real L state,
    /// matching the `state-derived` recurrence contract, but we still decode
    /// and validate the carrier so a corrupt/incompatible replay cannot hide in
    /// an otherwise apparently valid resume package.
    pub drift_state: HierarchosPortableReplayFloatTensor,
    pub rosa_token_histories: Vec<Vec<u32>>,
    pub ltm_state: HierarchosPortableLtmRunningState,
}

const PORTABLE_TRAINING_REPLAY_FORMAT: &str = "hierarchos-portable-training-replay-v1";

/// Encode JSON-native host metadata using the tagged, pickle-free replay ABI
/// shared with `tools.vulkan_optimizer_bridge`. Tensor-bearing values use the
/// specialized carrier codec below; this helper is deliberately limited to
/// JSON-native identity/configuration records.
pub fn encode_portable_replay_json(value: &serde_json::Value) -> Result<serde_json::Value> {
    Ok(match value {
        serde_json::Value::Array(items) => serde_json::json!({
            "__kind__": "list",
            "items": items
                .iter()
                .map(encode_portable_replay_json)
                .collect::<Result<Vec<_>>>()?,
        }),
        serde_json::Value::Object(items) => serde_json::json!({
            "__kind__": "dict",
            "items": items
                .iter()
                .map(|(key, value)| {
                    Ok(serde_json::json!([
                        key,
                        encode_portable_replay_json(value)?
                    ]))
                })
                .collect::<Result<Vec<_>>>()?,
        }),
        primitive => primitive.clone(),
    })
}

/// Decode JSON-native metadata from the portable replay ABI. Tensor refs and
/// backend-specific nodes are rejected here instead of being silently coerced;
/// callers that need those values must use their typed carrier decoder.
pub fn decode_portable_replay_json(value: &serde_json::Value) -> Result<serde_json::Value> {
    let Some(kind) = value.get("__kind__") else {
        return match value {
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                bail!("portable replay JSON container is missing __kind__")
            }
            primitive => Ok(primitive.clone()),
        };
    };
    let kind = kind
        .as_str()
        .context("portable replay JSON __kind__ must be a string")?;
    match kind {
        "list" | "tuple" => {
            let items = value
                .get("items")
                .and_then(serde_json::Value::as_array)
                .context("portable replay JSON sequence is missing items")?;
            Ok(serde_json::Value::Array(
                items
                    .iter()
                    .map(decode_portable_replay_json)
                    .collect::<Result<Vec<_>>>()?,
            ))
        }
        "dict" => {
            let items = value
                .get("items")
                .and_then(serde_json::Value::as_array)
                .context("portable replay JSON dictionary is missing items")?;
            let mut decoded = serde_json::Map::new();
            for pair in items {
                let pair = pair
                    .as_array()
                    .context("portable replay JSON dictionary entry is not a pair")?;
                if pair.len() != 2 {
                    bail!("portable replay JSON dictionary entry must contain two values");
                }
                let key = decode_portable_replay_json(&pair[0])?
                    .as_str()
                    .context("portable replay JSON dictionary key must decode to a string")?
                    .to_string();
                if decoded
                    .insert(key.clone(), decode_portable_replay_json(&pair[1])?)
                    .is_some()
                {
                    bail!("portable replay JSON dictionary contains duplicate key {key:?}");
                }
            }
            Ok(serde_json::Value::Object(decoded))
        }
        other => bail!("portable replay JSON metadata contains unsupported typed node {other:?}"),
    }
}

/// Read one JSON-native field from a portable replay document. This gives the
/// native trainer a typed, pickle-free identity boundary without duplicating
/// the Python replay parser or weakening the tensor-specific carrier checks.
pub fn read_portable_replay_json_field(
    replay_path: &Path,
    field: &str,
) -> Result<Option<serde_json::Value>> {
    if field.is_empty() {
        bail!("portable replay JSON field name must not be empty");
    }
    let document: serde_json::Value = serde_json::from_slice(
        &fs::read(replay_path).with_context(|| format!("reading {}", replay_path.display()))?,
    )
    .with_context(|| format!("decoding {}", replay_path.display()))?;
    let format = document
        .get("format")
        .and_then(serde_json::Value::as_str)
        .context("portable training replay is missing string format")?;
    if format != PORTABLE_TRAINING_REPLAY_FORMAT {
        bail!("unsupported portable training replay format {format:?}");
    }
    let state = document
        .get("state")
        .context("portable training replay is missing state")?;
    encoded_dict_get(state, field)?
        .map(decode_portable_replay_json)
        .transpose()
}

/// Decode the exact persisted recurrent carrier written by
/// `tools.vulkan_optimizer_bridge.write_vulkan_training_replay`.
///
/// Floating PyTorch tensors are widened to canonical FP32 on load. ROSA is
/// restored from either the serialized automaton list or the cached token
/// history tensor used by checkpoint-safe/cached-ID training. The automaton
/// form is validated through the native Rust ROSA implementation before its
/// bounded token history is accepted.
pub fn read_portable_running_carriers(
    replay_path: &Path,
    replay_tensor_path: Option<&Path>,
) -> Result<Option<HierarchosPortableRunningCarriers>> {
    let bytes =
        fs::read(replay_path).with_context(|| format!("reading {}", replay_path.display()))?;
    let document: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding {}", replay_path.display()))?;
    let format = document
        .get("format")
        .and_then(serde_json::Value::as_str)
        .context("portable training replay is missing string format")?;
    if format != PORTABLE_TRAINING_REPLAY_FORMAT {
        bail!("unsupported portable training replay format {format:?}");
    }
    let state = document
        .get("state")
        .context("portable training replay is missing state")?;
    let Some(running_states) = encoded_dict_get(state, "running_states")? else {
        return Ok(None);
    };
    let running = encoded_sequence_items(running_states, "running_states")?;
    if running.len() != 6 {
        bail!(
            "portable running_states must contain exactly six carriers; got {}",
            running.len()
        );
    }
    let tensor_path = replay_tensor_path.context(
        "portable running_states references tensors but the replay has no tensor sidecar",
    )?;
    let h_state = decode_float_tensor_ref(tensor_path, &running[0], "h_state")?;
    let l_state = decode_float_tensor_ref(tensor_path, &running[1], "l_state")?;
    let previous_context = decode_float_tensor_ref(tensor_path, &running[2], "prev_context")?;
    let target_context = decode_float_tensor_ref(tensor_path, &running[3], "target_context")?;
    let drift_state = decode_float_tensor_ref(tensor_path, &running[4], "drift_state")?;

    let ltm = encoded_sequence_items(&running[5], "running_states.ltm_state")?;
    if ltm.len() != 7 {
        bail!(
            "portable coherent-v9 LTM carrier must contain exactly seven entries; got {}",
            ltm.len()
        );
    }
    let fast_vals = decode_optional_float_tensor_ref(tensor_path, &ltm[0], "ltm.fast_vals")?;
    let mom_vals = decode_optional_float_tensor_ref(tensor_path, &ltm[1], "ltm._mom_vals")?;
    let rosa_token_histories = decode_rosa_histories(tensor_path, &ltm[2], &ltm[3])?;
    let timestamps = decode_optional_float_tensor_ref(tensor_path, &ltm[4], "ltm.timestamps")?;
    let sources = decode_optional_i64_tensor_ref(tensor_path, &ltm[5], "ltm.sources")?;
    let wallclock_timestamps =
        decode_optional_f64_tensor_ref(tensor_path, &ltm[6], "ltm.wallclock_timestamps")?;

    Ok(Some(HierarchosPortableRunningCarriers {
        h_state,
        l_state,
        previous_context,
        target_context,
        drift_state,
        rosa_token_histories,
        ltm_state: HierarchosPortableLtmRunningState {
            fast_vals,
            mom_vals,
            timestamps,
            sources,
            wallclock_timestamps,
        },
    }))
}

/// Materialize the PyTorch-compatible LTM continuation state for a fresh
/// native run from the canonical model package. The learned/consolidated
/// Vulkan training path currently treats these transient values as read-only.
/// Older inference-oriented packages may omit zero-initialized metadata; those
/// fields are reconstructed using the exact PyTorch `LTMModule` defaults.
pub fn read_model_ltm_running_state(
    model_path: &Path,
) -> Result<HierarchosPortableLtmRunningState> {
    let (fast_shape, fast_values) = read_f32_tensor(model_path, "ltm.fast_vals")?;
    if fast_shape.len() != 2 {
        bail!("ltm.fast_vals must have [slots, value_dim] geometry, got {fast_shape:?}");
    }
    let slots = fast_shape[0];

    let (mom_shape, mom_values) = read_optional_f32_tensor(model_path, "ltm._mom_vals")?
        .unwrap_or_else(|| (fast_shape.clone(), vec![0.0; fast_values.len()]));
    if mom_shape != fast_shape {
        bail!("ltm._mom_vals shape {mom_shape:?} does not match ltm.fast_vals {fast_shape:?}");
    }

    let timestamp_shape = vec![slots];
    let (timestamps_shape, timestamps_values) =
        read_optional_f32_tensor(model_path, "ltm.timestamps")?
            .unwrap_or_else(|| (timestamp_shape.clone(), vec![0.0; slots]));
    if timestamps_shape != timestamp_shape {
        bail!(
            "ltm.timestamps shape {timestamps_shape:?} does not match expected {timestamp_shape:?}"
        );
    }

    let (sources_shape, sources_values) = read_optional_i64_tensor(model_path, "ltm.sources")?
        .unwrap_or_else(|| (timestamp_shape.clone(), vec![0; slots]));
    if sources_shape != timestamp_shape {
        bail!("ltm.sources shape {sources_shape:?} does not match expected {timestamp_shape:?}");
    }

    let (wallclock_shape, wallclock_values) =
        read_optional_f64_tensor(model_path, "ltm.wallclock_timestamps")?
            .unwrap_or_else(|| (timestamp_shape.clone(), vec![0.0; slots]));
    if wallclock_shape != timestamp_shape {
        bail!(
            "ltm.wallclock_timestamps shape {wallclock_shape:?} does not match expected {timestamp_shape:?}"
        );
    }

    Ok(HierarchosPortableLtmRunningState {
        fast_vals: Some(HierarchosPortableReplayFloatTensor {
            shape: fast_shape,
            values: fast_values,
        }),
        mom_vals: Some(HierarchosPortableReplayFloatTensor {
            shape: mom_shape,
            values: mom_values,
        }),
        timestamps: Some(HierarchosPortableReplayFloatTensor {
            shape: timestamps_shape,
            values: timestamps_values,
        }),
        sources: Some(HierarchosPortableReplayI64Tensor {
            shape: sources_shape,
            values: sources_values,
        }),
        wallclock_timestamps: Some(HierarchosPortableReplayF64Tensor {
            shape: wallclock_shape,
            values: wallclock_values,
        }),
    })
}

/// Encode a complete exact-resume `running_states` value plus its tensor
/// sidecar payload using the same JSON grammar as the Python bridge.
///
/// ROSA automata are emitted as host JSON rather than a rectangular token
/// tensor so lanes with different bounded-history lengths remain lossless.
pub fn encode_portable_running_carriers(
    carriers: &HierarchosPortableRunningCarriers,
) -> Result<(serde_json::Value, Vec<HierarchosPortableReplayTensor>)> {
    validate_replay_float_tensor("h_state", &carriers.h_state)?;
    validate_replay_float_tensor("l_state", &carriers.l_state)?;
    validate_replay_float_tensor("prev_context", &carriers.previous_context)?;
    validate_replay_float_tensor("target_context", &carriers.target_context)?;
    validate_replay_float_tensor("drift_state", &carriers.drift_state)?;
    if let Some(value) = carriers.ltm_state.fast_vals.as_ref() {
        validate_replay_float_tensor("ltm.fast_vals", value)?;
    }
    if let Some(value) = carriers.ltm_state.mom_vals.as_ref() {
        validate_replay_float_tensor("ltm._mom_vals", value)?;
    }
    if let Some(value) = carriers.ltm_state.timestamps.as_ref() {
        validate_replay_float_tensor("ltm.timestamps", value)?;
    }
    if let Some(value) = carriers.ltm_state.sources.as_ref() {
        validate_replay_i64_tensor("ltm.sources", value)?;
    }
    if let Some(value) = carriers.ltm_state.wallclock_timestamps.as_ref() {
        validate_replay_f64_tensor("ltm.wallclock_timestamps", value)?;
    }
    if carriers.rosa_token_histories.is_empty() {
        bail!("portable running state must contain at least one ROSA lane history");
    }

    const H_STATE: &str = "running_h_state";
    const L_STATE: &str = "running_l_state";
    const PREV_CONTEXT: &str = "running_prev_context";
    const TARGET_CONTEXT: &str = "running_target_context";
    const DRIFT_STATE: &str = "running_drift_state";
    const LTM_FAST: &str = "running_ltm_fast_vals";
    const LTM_MOM: &str = "running_ltm_mom_vals";
    const LTM_TIMESTAMPS: &str = "running_ltm_timestamps";
    const LTM_SOURCES: &str = "running_ltm_sources";
    const LTM_WALLCLOCK: &str = "running_ltm_wallclock_timestamps";

    let mut tensors = vec![
        HierarchosPortableReplayTensor::f32(
            H_STATE,
            carriers.h_state.shape.clone(),
            carriers.h_state.values.clone(),
        ),
        HierarchosPortableReplayTensor::f32(
            L_STATE,
            carriers.l_state.shape.clone(),
            carriers.l_state.values.clone(),
        ),
        HierarchosPortableReplayTensor::f32(
            PREV_CONTEXT,
            carriers.previous_context.shape.clone(),
            carriers.previous_context.values.clone(),
        ),
        HierarchosPortableReplayTensor::f32(
            TARGET_CONTEXT,
            carriers.target_context.shape.clone(),
            carriers.target_context.values.clone(),
        ),
        HierarchosPortableReplayTensor::f32(
            DRIFT_STATE,
            carriers.drift_state.shape.clone(),
            carriers.drift_state.values.clone(),
        ),
    ];
    if let Some(value) = carriers.ltm_state.fast_vals.as_ref() {
        tensors.push(HierarchosPortableReplayTensor::f32(
            LTM_FAST,
            value.shape.clone(),
            value.values.clone(),
        ));
    }
    if let Some(value) = carriers.ltm_state.mom_vals.as_ref() {
        tensors.push(HierarchosPortableReplayTensor::f32(
            LTM_MOM,
            value.shape.clone(),
            value.values.clone(),
        ));
    }
    if let Some(value) = carriers.ltm_state.timestamps.as_ref() {
        tensors.push(HierarchosPortableReplayTensor::f32(
            LTM_TIMESTAMPS,
            value.shape.clone(),
            value.values.clone(),
        ));
    }
    if let Some(value) = carriers.ltm_state.sources.as_ref() {
        tensors.push(HierarchosPortableReplayTensor::i64(
            LTM_SOURCES,
            value.shape.clone(),
            value.values.clone(),
        ));
    }
    if let Some(value) = carriers.ltm_state.wallclock_timestamps.as_ref() {
        tensors.push(HierarchosPortableReplayTensor::f64(
            LTM_WALLCLOCK,
            value.shape.clone(),
            value.values.clone(),
        ));
    }
    let tensor_ref = |name: &str| serde_json::json!({"__kind__": "tensor", "name": name});
    let optional_tensor_ref = |name: &str, present: bool| {
        if present {
            tensor_ref(name)
        } else {
            serde_json::Value::Null
        }
    };
    let rosa_states = carriers
        .rosa_token_histories
        .iter()
        .map(|history| encode_rosa_state(history))
        .collect::<Result<Vec<_>>>()?;
    let encoded = serde_json::json!({
        "__kind__": "tuple",
        "items": [
            tensor_ref(H_STATE),
            tensor_ref(L_STATE),
            tensor_ref(PREV_CONTEXT),
            tensor_ref(TARGET_CONTEXT),
            tensor_ref(DRIFT_STATE),
            {
                "__kind__": "tuple",
                "items": [
                    optional_tensor_ref(LTM_FAST, carriers.ltm_state.fast_vals.is_some()),
                    optional_tensor_ref(LTM_MOM, carriers.ltm_state.mom_vals.is_some()),
                    null,
                    {"__kind__": "list", "items": rosa_states},
                    optional_tensor_ref(LTM_TIMESTAMPS, carriers.ltm_state.timestamps.is_some()),
                    optional_tensor_ref(LTM_SOURCES, carriers.ltm_state.sources.is_some()),
                    optional_tensor_ref(LTM_WALLCLOCK, carriers.ltm_state.wallclock_timestamps.is_some())
                ]
            }
        ]
    });
    Ok((encoded, tensors))
}

fn encode_rosa_state(history: &[u32]) -> Result<serde_json::Value> {
    let mut state = RosaState::new();
    for &token in history {
        state.predict_and_push(token, 0);
    }
    let snapshot = state.snapshot();
    let transitions = snapshot
        .transitions
        .iter()
        .enumerate()
        .map(|(source, row)| {
            serde_json::json!([
                source,
                row.iter()
                    .map(|edge| serde_json::json!([edge.symbol, edge.target]))
                    .collect::<Vec<_>>()
            ])
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "__kind__": "rosa_state",
        "transitions": transitions,
        "suffix_links": snapshot.suffix_links,
        "lengths": snapshot.lengths,
        "endpos": snapshot.endpos,
        "last_state": snapshot.last_state,
        "num_states": snapshot.transitions.len(),
        "tokens": snapshot.tokens,
    }))
}

fn encoded_dict_get<'a>(
    value: &'a serde_json::Value,
    wanted: &str,
) -> Result<Option<&'a serde_json::Value>> {
    if value.get("__kind__").and_then(serde_json::Value::as_str) != Some("dict") {
        bail!("portable replay node for dictionary lookup is not __kind__=dict");
    }
    let items = value
        .get("items")
        .and_then(serde_json::Value::as_array)
        .context("portable replay dict node is missing items array")?;
    let mut found = None;
    for pair in items {
        let pair = pair
            .as_array()
            .context("portable replay dict item is not a key/value pair")?;
        if pair.len() != 2 {
            bail!("portable replay dict item must contain exactly two values");
        }
        if pair[0].as_str() == Some(wanted) {
            if found.is_some() {
                bail!("portable replay dict contains duplicate key {wanted:?}");
            }
            found = Some(&pair[1]);
        }
    }
    Ok(found)
}

fn encoded_sequence_items<'a>(
    value: &'a serde_json::Value,
    label: &str,
) -> Result<&'a Vec<serde_json::Value>> {
    let kind = value
        .get("__kind__")
        .and_then(serde_json::Value::as_str)
        .with_context(|| format!("portable {label} is missing __kind__"))?;
    if kind != "tuple" && kind != "list" {
        bail!("portable {label} must be an encoded tuple/list, got {kind:?}");
    }
    value
        .get("items")
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("portable {label} is missing items array"))
}

fn tensor_ref_name<'a>(value: &'a serde_json::Value, label: &str) -> Result<&'a str> {
    if value.get("__kind__").and_then(serde_json::Value::as_str) != Some("tensor") {
        bail!("portable {label} must be a tensor reference");
    }
    value
        .get("name")
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .with_context(|| format!("portable {label} tensor reference has no name"))
}

fn decode_float_tensor_ref(
    tensor_path: &Path,
    value: &serde_json::Value,
    label: &str,
) -> Result<HierarchosPortableReplayFloatTensor> {
    let name = tensor_ref_name(value, label)?;
    let (shape, values) = read_f32_tensor(tensor_path, name)
        .with_context(|| format!("decoding portable {label} tensor {name:?}"))?;
    Ok(HierarchosPortableReplayFloatTensor { shape, values })
}

fn decode_optional_float_tensor_ref(
    tensor_path: &Path,
    value: &serde_json::Value,
    label: &str,
) -> Result<Option<HierarchosPortableReplayFloatTensor>> {
    if value.is_null() {
        return Ok(None);
    }
    decode_float_tensor_ref(tensor_path, value, label).map(Some)
}

fn decode_f64_tensor_ref(
    tensor_path: &Path,
    value: &serde_json::Value,
    label: &str,
) -> Result<HierarchosPortableReplayF64Tensor> {
    let name = tensor_ref_name(value, label)?;
    let (shape, values) = read_f64_tensor(tensor_path, name)
        .with_context(|| format!("decoding portable {label} tensor {name:?}"))?;
    Ok(HierarchosPortableReplayF64Tensor { shape, values })
}

fn decode_optional_f64_tensor_ref(
    tensor_path: &Path,
    value: &serde_json::Value,
    label: &str,
) -> Result<Option<HierarchosPortableReplayF64Tensor>> {
    if value.is_null() {
        return Ok(None);
    }
    decode_f64_tensor_ref(tensor_path, value, label).map(Some)
}

fn decode_i64_tensor_ref(
    tensor_path: &Path,
    value: &serde_json::Value,
    label: &str,
) -> Result<HierarchosPortableReplayI64Tensor> {
    let name = tensor_ref_name(value, label)?;
    let (shape, values) = read_i64_tensor(tensor_path, name)
        .with_context(|| format!("decoding portable {label} tensor {name:?}"))?;
    Ok(HierarchosPortableReplayI64Tensor { shape, values })
}

fn decode_optional_i64_tensor_ref(
    tensor_path: &Path,
    value: &serde_json::Value,
    label: &str,
) -> Result<Option<HierarchosPortableReplayI64Tensor>> {
    if value.is_null() {
        return Ok(None);
    }
    decode_i64_tensor_ref(tensor_path, value, label).map(Some)
}

fn validate_replay_float_tensor(
    name: &str,
    tensor: &HierarchosPortableReplayFloatTensor,
) -> Result<()> {
    validate_f32_payload(name, &tensor.shape, &tensor.values)
}

fn validate_replay_f64_tensor(
    name: &str,
    tensor: &HierarchosPortableReplayF64Tensor,
) -> Result<()> {
    let expected = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .context("tensor element count overflow")?;
    if expected != tensor.values.len() {
        bail!(
            "tensor {name:?} shape {:?} expects {expected} values, got {}",
            tensor.shape,
            tensor.values.len()
        );
    }
    if tensor.values.iter().any(|value| !value.is_finite()) {
        bail!("tensor {name:?} contains non-finite values");
    }
    Ok(())
}

fn validate_replay_i64_tensor(
    name: &str,
    tensor: &HierarchosPortableReplayI64Tensor,
) -> Result<()> {
    let expected = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .context("tensor element count overflow")?;
    if expected != tensor.values.len() {
        bail!(
            "tensor {name:?} shape {:?} expects {expected} values, got {}",
            tensor.shape,
            tensor.values.len()
        );
    }
    Ok(())
}

fn decode_rosa_histories(
    tensor_path: &Path,
    encoded_past_tokens: &serde_json::Value,
    encoded_rosa_states: &serde_json::Value,
) -> Result<Vec<Vec<u32>>> {
    if !encoded_rosa_states.is_null() {
        let states = encoded_sequence_items(encoded_rosa_states, "ROSA state list")?;
        if !states.is_empty() && states.iter().all(|state| !state.is_null()) {
            return states
                .iter()
                .enumerate()
                .map(|(row, state)| decode_rosa_state_tokens(state, row))
                .collect();
        }
    }

    if !encoded_past_tokens.is_null() {
        let name = tensor_ref_name(encoded_past_tokens, "ROSA past_tokens")?;
        let (shape, values) = read_u32_tensor(tensor_path, name)
            .with_context(|| format!("decoding portable ROSA token history {name:?}"))?;
        return rows_from_token_tensor(&shape, &values);
    }

    bail!(
        "portable coherent-v9 LTM carrier has neither complete ROSA automata nor cached token history"
    )
}

fn rows_from_token_tensor(shape: &[usize], values: &[u32]) -> Result<Vec<Vec<u32>>> {
    match shape {
        [tokens] => {
            if *tokens != values.len() {
                bail!("ROSA token tensor shape does not match payload length");
            }
            Ok(vec![values.to_vec()])
        }
        [rows, tokens] => {
            let expected = rows
                .checked_mul(*tokens)
                .context("ROSA token tensor shape overflow")?;
            if expected != values.len() {
                bail!("ROSA token tensor shape does not match payload length");
            }
            Ok(values.chunks_exact(*tokens).map(<[u32]>::to_vec).collect())
        }
        _ => bail!(
            "ROSA cached token history must have shape [tokens] or [batch,tokens], got {shape:?}"
        ),
    }
}

fn decode_rosa_state_tokens(value: &serde_json::Value, row: usize) -> Result<Vec<u32>> {
    if value.get("__kind__").and_then(serde_json::Value::as_str) != Some("rosa_state") {
        bail!("portable ROSA state row {row} is not __kind__=rosa_state");
    }
    let num_states = json_usize(
        value
            .get("num_states")
            .with_context(|| format!("ROSA state row {row} is missing num_states"))?,
        "ROSA num_states",
    )?;
    if num_states == 0 {
        bail!("portable ROSA state row {row} has zero states");
    }
    let encoded_transitions = value
        .get("transitions")
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("ROSA state row {row} is missing transitions"))?;
    let mut transitions = vec![Vec::new(); num_states];
    let mut transition_rows_seen = vec![false; num_states];
    for encoded_row in encoded_transitions {
        let encoded_row = encoded_row
            .as_array()
            .context("ROSA transition row is not a pair")?;
        if encoded_row.len() != 2 {
            bail!("ROSA transition row must contain state index and transitions");
        }
        let source = json_usize(&encoded_row[0], "ROSA transition source")?;
        if source >= num_states || transition_rows_seen[source] {
            bail!("ROSA transition table has invalid/duplicate source state {source}");
        }
        transition_rows_seen[source] = true;
        let entries = encoded_row[1]
            .as_array()
            .context("ROSA transition entries are not an array")?;
        let mut decoded = Vec::with_capacity(entries.len());
        for entry in entries {
            let entry = entry.as_array().context("ROSA transition is not a pair")?;
            if entry.len() != 2 {
                bail!("ROSA transition must contain symbol and target");
            }
            let symbol = json_u32(&entry[0], "ROSA transition symbol")?;
            let target = json_usize(&entry[1], "ROSA transition target")?;
            decoded.push(RosaTransitionSnapshot { symbol, target });
        }
        transitions[source] = decoded;
    }
    if transition_rows_seen.iter().any(|seen| !seen) {
        bail!("ROSA transition table omits one or more states");
    }

    let suffix_links = json_isize_vec(value, "suffix_links", row)?;
    let lengths = json_usize_vec(value, "lengths", row)?;
    let endpos = json_isize_vec(value, "endpos", row)?;
    let last_state = json_usize(
        value
            .get("last_state")
            .with_context(|| format!("ROSA state row {row} is missing last_state"))?,
        "ROSA last_state",
    )?;
    let tokens = value
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("ROSA state row {row} is missing tokens"))?
        .iter()
        .map(|value| json_u32(value, "ROSA token"))
        .collect::<Result<Vec<_>>>()?;
    let snapshot = RosaStateSnapshot {
        transitions,
        suffix_links,
        lengths,
        endpos,
        last_state,
        tokens,
    };
    if snapshot.transitions.len() != num_states {
        bail!("ROSA state row {row} num_states disagrees with transition table");
    }
    let validated = RosaState::from_snapshot(&snapshot)
        .with_context(|| format!("validating portable ROSA state row {row}"))?;
    Ok(validated.snapshot().tokens)
}

fn json_usize(value: &serde_json::Value, label: &str) -> Result<usize> {
    let value = value
        .as_u64()
        .with_context(|| format!("{label} must be a non-negative integer"))?;
    usize::try_from(value).with_context(|| format!("{label} exceeds host usize range"))
}

fn json_u32(value: &serde_json::Value, label: &str) -> Result<u32> {
    let value = value
        .as_u64()
        .with_context(|| format!("{label} must be a non-negative integer"))?;
    u32::try_from(value).with_context(|| format!("{label} exceeds u32 range"))
}

fn json_isize(value: &serde_json::Value, label: &str) -> Result<isize> {
    let value = value
        .as_i64()
        .with_context(|| format!("{label} must be an integer"))?;
    isize::try_from(value).with_context(|| format!("{label} exceeds host isize range"))
}

fn json_usize_vec(value: &serde_json::Value, key: &str, row: usize) -> Result<Vec<usize>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("ROSA state row {row} is missing {key}"))?
        .iter()
        .map(|value| json_usize(value, key))
        .collect()
}

fn json_isize_vec(value: &serde_json::Value, key: &str, row: usize) -> Result<Vec<isize>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("ROSA state row {row} is missing {key}"))?
        .iter()
        .map(|value| json_isize(value, key))
        .collect()
}

impl HierarchosPortableReplayTensor {
    pub fn u8(name: impl Into<String>, shape: Vec<usize>, values: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: HierarchosPortableReplayTensorData::U8(values),
        }
    }

    pub fn f32(name: impl Into<String>, shape: Vec<usize>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: HierarchosPortableReplayTensorData::F32(values),
        }
    }

    pub fn f64(name: impl Into<String>, shape: Vec<usize>, values: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: HierarchosPortableReplayTensorData::F64(values),
        }
    }

    pub fn i64(name: impl Into<String>, shape: Vec<usize>, values: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: HierarchosPortableReplayTensorData::I64(values),
        }
    }
}

/// Write the tensor half of the cross-backend replay wire format consumed by
/// `tools.vulkan_optimizer_bridge.read_vulkan_training_replay`.
pub fn write_portable_replay_tensors(
    path: &Path,
    format: &str,
    tensors: &[HierarchosPortableReplayTensor],
) -> Result<()> {
    if tensors.is_empty() {
        bail!("portable replay tensor checkpoint requires at least one tensor");
    }

    struct OwnedTensor {
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    let mut seen = std::collections::HashSet::new();
    let mut owned = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        if tensor.name.trim().is_empty() || !seen.insert(tensor.name.as_str()) {
            bail!(
                "portable replay tensor checkpoint has empty/duplicate tensor {:?}",
                tensor.name
            );
        }
        let element_count = tensor.shape.iter().try_fold(1usize, |count, dim| {
            count
                .checked_mul(*dim)
                .context("portable replay tensor shape element count overflow")
        })?;
        let (dtype, value_count, bytes) = match &tensor.data {
            HierarchosPortableReplayTensorData::U8(values) => {
                (Dtype::U8, values.len(), values.clone())
            }
            HierarchosPortableReplayTensorData::F32(values) => {
                if values.iter().any(|value| !value.is_finite()) {
                    bail!(
                        "portable replay tensor {:?} contains non-finite FP32 values",
                        tensor.name
                    );
                }
                (Dtype::F32, values.len(), f32_bytes(values))
            }
            HierarchosPortableReplayTensorData::F64(values) => {
                if values.iter().any(|value| !value.is_finite()) {
                    bail!(
                        "portable replay tensor {:?} contains non-finite FP64 values",
                        tensor.name
                    );
                }
                (Dtype::F64, values.len(), f64_bytes(values))
            }
            HierarchosPortableReplayTensorData::I64(values) => {
                (Dtype::I64, values.len(), i64_bytes(values))
            }
        };
        if element_count != value_count {
            bail!(
                "portable replay tensor {:?} shape {:?} contains {} elements but payload has {}",
                tensor.name,
                tensor.shape,
                element_count,
                value_count
            );
        }
        owned.push(OwnedTensor {
            name: tensor.name.clone(),
            dtype,
            shape: tensor.shape.clone(),
            bytes,
        });
    }

    let mut views = Vec::with_capacity(owned.len());
    for tensor in &owned {
        views.push((
            tensor.name.as_str(),
            TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes)?,
        ));
    }
    let metadata = HashMap::from([("format".to_string(), format.to_string())]);
    serialize_to_file(views, Some(metadata), path)?;
    Ok(())
}

pub fn read_f32_tensor(path: &Path, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?} in {}", path.display()))?;
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
        bail!(
            "tensor {name:?} contains non-finite {:?} values",
            tensor.dtype()
        );
    }
    Ok((tensor.shape().to_vec(), values))
}

fn read_f64_tensor(path: &Path, name: &str) -> Result<(Vec<usize>, Vec<f64>)> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?} in {}", path.display()))?;
    let raw = tensor.data();
    let mut values = Vec::with_capacity(match tensor.dtype() {
        Dtype::F64 => raw.len() / 8,
        Dtype::F32 => raw.len() / 4,
        Dtype::F16 | Dtype::BF16 => raw.len() / 2,
        dtype => bail!("tensor {name:?} must be F64/F32/F16/BF16, got {dtype:?}"),
    });
    match tensor.dtype() {
        Dtype::F64 => {
            if raw.len() % 8 != 0 {
                bail!("tensor {name:?} has invalid FP64 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(8) {
                values.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
        }
        Dtype::F32 => {
            if raw.len() % 4 != 0 {
                bail!("tensor {name:?} has invalid FP32 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(4) {
                values.push(f64::from(f32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        Dtype::F16 => {
            if raw.len() % 2 != 0 {
                bail!("tensor {name:?} has invalid FP16 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(2) {
                values.push(f64::from(f16_bits_to_f32(u16::from_le_bytes([
                    chunk[0], chunk[1],
                ]))));
            }
        }
        Dtype::BF16 => {
            if raw.len() % 2 != 0 {
                bail!("tensor {name:?} has invalid BF16 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(2) {
                values.push(f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16,
                )));
            }
        }
        _ => unreachable!("dtype validated above"),
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!(
            "tensor {name:?} contains non-finite {:?} values",
            tensor.dtype()
        );
    }
    Ok((tensor.shape().to_vec(), values))
}

fn read_i64_tensor(path: &Path, name: &str) -> Result<(Vec<usize>, Vec<i64>)> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?} in {}", path.display()))?;
    let raw = tensor.data();
    let mut values = Vec::with_capacity(match tensor.dtype() {
        Dtype::U8 => raw.len(),
        Dtype::U32 | Dtype::I32 => raw.len() / 4,
        Dtype::I64 => raw.len() / 8,
        dtype => bail!("integer tensor {name:?} must be U8/U32/I32/I64, got {dtype:?}"),
    });
    match tensor.dtype() {
        Dtype::U8 => values.extend(raw.iter().copied().map(i64::from)),
        Dtype::U32 => {
            if raw.len() % 4 != 0 {
                bail!(
                    "integer tensor {name:?} has invalid U32 byte length {}",
                    raw.len()
                );
            }
            for chunk in raw.chunks_exact(4) {
                values.push(i64::from(u32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        Dtype::I32 => {
            if raw.len() % 4 != 0 {
                bail!(
                    "integer tensor {name:?} has invalid I32 byte length {}",
                    raw.len()
                );
            }
            for chunk in raw.chunks_exact(4) {
                values.push(i64::from(i32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        Dtype::I64 => {
            if raw.len() % 8 != 0 {
                bail!(
                    "integer tensor {name:?} has invalid I64 byte length {}",
                    raw.len()
                );
            }
            for chunk in raw.chunks_exact(8) {
                values.push(i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
        }
        _ => unreachable!("dtype validated above"),
    }
    Ok((tensor.shape().to_vec(), values))
}

fn tensor_exists(path: &Path, name: &str) -> Result<bool> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    Ok(tensors.tensor(name).is_ok())
}

fn read_optional_f32_tensor(path: &Path, name: &str) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
    if !tensor_exists(path, name)? {
        return Ok(None);
    }
    read_f32_tensor(path, name).map(Some)
}

fn read_optional_f64_tensor(path: &Path, name: &str) -> Result<Option<(Vec<usize>, Vec<f64>)>> {
    if !tensor_exists(path, name)? {
        return Ok(None);
    }
    read_f64_tensor(path, name).map(Some)
}

fn read_optional_i64_tensor(path: &Path, name: &str) -> Result<Option<(Vec<usize>, Vec<i64>)>> {
    if !tensor_exists(path, name)? {
        return Ok(None);
    }
    read_i64_tensor(path, name).map(Some)
}

fn read_u32_tensor(path: &Path, name: &str) -> Result<(Vec<usize>, Vec<u32>)> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?} in {}", path.display()))?;
    let raw = tensor.data();
    let mut values = Vec::with_capacity(match tensor.dtype() {
        Dtype::U8 => raw.len(),
        Dtype::U32 | Dtype::I32 => raw.len() / 4,
        Dtype::I64 => raw.len() / 8,
        dtype => bail!("token tensor {name:?} must be U8/U32/I32/I64, got {dtype:?}"),
    });
    match tensor.dtype() {
        Dtype::U8 => values.extend(raw.iter().copied().map(u32::from)),
        Dtype::U32 => {
            if raw.len() % 4 != 0 {
                bail!(
                    "token tensor {name:?} has invalid U32 byte length {}",
                    raw.len()
                );
            }
            for chunk in raw.chunks_exact(4) {
                values.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Dtype::I32 => {
            if raw.len() % 4 != 0 {
                bail!(
                    "token tensor {name:?} has invalid I32 byte length {}",
                    raw.len()
                );
            }
            for chunk in raw.chunks_exact(4) {
                let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                values.push(u32::try_from(value).with_context(|| {
                    format!("token tensor {name:?} contains negative value {value}")
                })?);
            }
        }
        Dtype::I64 => {
            if raw.len() % 8 != 0 {
                bail!(
                    "token tensor {name:?} has invalid I64 byte length {}",
                    raw.len()
                );
            }
            for chunk in raw.chunks_exact(8) {
                let value = i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                values.push(u32::try_from(value).with_context(|| {
                    format!("token tensor {name:?} contains out-of-range token {value}")
                })?);
            }
        }
        _ => unreachable!("dtype validated above"),
    }
    Ok((tensor.shape().to_vec(), values))
}

/// Verify that the named optimizer-bound model tensors are physically stored as
/// canonical FP32 masters. Lower-precision model input is accepted by ordinary
/// inference/model loading, but a portable training package that declares the
/// FP32-master ABI must never persist F16/BF16 execution storage as authority.
pub fn validate_f32_tensor_names(path: &Path, names: &[&str]) -> Result<()> {
    if names.is_empty() {
        bail!("portable FP32 master validation requires at least one tensor name");
    }
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut seen = std::collections::HashSet::new();
    for &name in names {
        if name.trim().is_empty() || !seen.insert(name) {
            bail!("portable FP32 master validation has empty/duplicate tensor {name:?}");
        }
        let tensor = tensors.tensor(name).with_context(|| {
            format!(
                "portable FP32 master file {} is missing optimizer slot {name:?}",
                path.display()
            )
        })?;
        if tensor.dtype() != Dtype::F32 {
            bail!(
                "portable FP32 master slot {name:?} is stored as {:?} in {}; lower-precision storage is reserved for derived execution mirrors",
                tensor.dtype(),
                path.display()
            );
        }
    }
    Ok(())
}

pub fn write_f32_tensor(path: &Path, name: &str, shape: &[usize], values: &[f32]) -> Result<()> {
    validate_f32_payload(name, shape, values)?;
    let bytes = f32_bytes(values);
    let view = TensorView::new(Dtype::F32, shape.to_vec(), &bytes)?;
    let metadata = HashMap::from([
        (
            "format".to_string(),
            "hierarchos-vulkan-fp32-v1".to_string(),
        ),
        ("layout".to_string(), "pytorch-row-major".to_string()),
    ]);
    serialize_to_file([(name, view)], Some(metadata), path)?;
    Ok(())
}

/// Rebuild a SafeTensors file while replacing one trainable floating tensor with
/// its canonical FP32 master value. F32/F16/BF16 source storage is accepted;
/// every other tensor and the file metadata are preserved byte-for-byte.
pub fn replace_f32_tensor(
    source: &Path,
    destination: &Path,
    name: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<()> {
    replace_f32_tensors(source, destination, &[(name, shape, values)])
}

/// Rebuild a SafeTensors file while replacing floating model tensors by name and
/// keeping each source tensor's existing shape. Replacements are written as
/// canonical FP32 masters even when the source tensor was F16/BF16. This is the
/// efficient full-model checkpoint seam: the source file is inspected once for
/// shapes, then rebuilt once, rather than reparsing a multi-gigabyte model
/// separately per tensor.
pub fn replace_f32_tensor_values(
    source: &Path,
    destination: &Path,
    replacements: &[(&str, &[f32])],
) -> Result<()> {
    if replacements.is_empty() {
        bail!("at least one SafeTensors replacement is required");
    }
    let source_bytes = fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let tensors = SafeTensors::deserialize(&source_bytes)?;
    let mut shapes = Vec::with_capacity(replacements.len());
    for (index, (name, values)) in replacements.iter().enumerate() {
        if replacements[..index]
            .iter()
            .any(|(previous, _)| previous == name)
        {
            bail!("duplicate SafeTensors replacement for tensor {name:?}");
        }
        let tensor = tensors.tensor(name).with_context(|| {
            format!(
                "missing tensor {name:?} in source model {}",
                source.display()
            )
        })?;
        if !is_supported_model_float_dtype(tensor.dtype()) {
            bail!(
                "tensor {name:?} must be F32/F16/BF16, got {:?}",
                tensor.dtype()
            );
        }
        let shape = tensor.shape().to_vec();
        validate_f32_payload(name, &shape, values)?;
        shapes.push(shape);
    }
    drop(tensors);
    drop(source_bytes);
    let shaped = replacements
        .iter()
        .zip(&shapes)
        .map(|((name, values), shape)| (*name, shape.as_slice(), *values))
        .collect::<Vec<_>>();
    replace_f32_tensors(source, destination, &shaped)
}

/// Rebuild a SafeTensors file while atomically replacing several trainable model
/// tensors with canonical FP32 masters. Tensor names, shapes, dtypes, and
/// metadata for every untouched parameter are preserved exactly, keeping mixed
/// F32/F16/BF16 Vulkan checkpoints interchangeable with PyTorch and the native
/// Hierarchos inference runtime.
pub fn replace_f32_tensors(
    source: &Path,
    destination: &Path,
    replacements: &[(&str, &[usize], &[f32])],
) -> Result<()> {
    if replacements.is_empty() {
        bail!("at least one SafeTensors replacement is required");
    }
    for (index, (name, shape, values)) in replacements.iter().enumerate() {
        validate_f32_payload(name, shape, values)?;
        if replacements[..index]
            .iter()
            .any(|(previous, _, _)| previous == name)
        {
            bail!("duplicate SafeTensors replacement for tensor {name:?}");
        }
    }
    let source_bytes = fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let tensors = SafeTensors::deserialize(&source_bytes)?;
    let (_, header) = SafeTensors::read_metadata(&source_bytes)?;
    let metadata = header.metadata().clone();
    let replacement_bytes: Vec<Vec<u8>> = replacements
        .iter()
        .map(|(_, _, values)| f32_bytes(values))
        .collect();

    struct OwnedTensor {
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    let mut found = vec![false; replacements.len()];
    let mut owned = Vec::with_capacity(tensors.len());
    for tensor_name in tensors.names() {
        let tensor = tensors.tensor(tensor_name)?;
        if let Some((replacement_index, (_, shape, _))) = replacements
            .iter()
            .enumerate()
            .find(|(_, (name, _, _))| tensor_name == *name)
        {
            if !is_supported_model_float_dtype(tensor.dtype()) {
                bail!(
                    "tensor {tensor_name:?} must be F32/F16/BF16, got {:?}",
                    tensor.dtype()
                );
            }
            if tensor.shape() != *shape {
                bail!(
                    "tensor {tensor_name:?} has source shape {:?}; replacement shape is {shape:?}",
                    tensor.shape()
                );
            }
            owned.push(OwnedTensor {
                name: tensor_name.to_string(),
                dtype: Dtype::F32,
                shape: shape.to_vec(),
                data: replacement_bytes[replacement_index].clone(),
            });
            found[replacement_index] = true;
        } else {
            owned.push(OwnedTensor {
                name: tensor_name.to_string(),
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                data: tensor.data().to_vec(),
            });
        }
    }
    for (index, is_found) in found.iter().enumerate() {
        if !is_found {
            bail!(
                "source checkpoint does not contain tensor {:?}",
                replacements[index].0
            );
        }
    }

    let mut views = Vec::with_capacity(owned.len());
    for tensor in &owned {
        views.push((
            tensor.name.as_str(),
            TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)?,
        ));
    }
    serialize_to_file(views, metadata, destination)?;
    Ok(())
}

/// Write a PyTorch-readable SafeTensors companion checkpoint containing AdamW
/// moment state. Model parameters stay in `model.safetensors`; keeping the
/// optimizer separate means the model package remains inference-compatible
/// with native Rust, PyTorch CPU, and PyTorch CUDA without training-only keys.
pub fn write_adamw_optimizer_state(path: &Path, state: &AdamWOptimizerState) -> Result<()> {
    if state.slots.is_empty() {
        bail!("AdamW optimizer checkpoint requires at least one slot");
    }
    let mut seen = std::collections::HashSet::new();
    let mut owned = Vec::with_capacity(state.slots.len() * 2);
    for slot in &state.slots {
        if slot.name.trim().is_empty() || !seen.insert(slot.name.as_str()) {
            bail!(
                "AdamW optimizer checkpoint has empty/duplicate slot {:?}",
                slot.name
            );
        }
        if slot.exp_avg.is_empty() || slot.exp_avg.len() != slot.exp_avg_sq.len() {
            bail!(
                "AdamW optimizer slot {:?} has incompatible moment lengths {}/{}",
                slot.name,
                slot.exp_avg.len(),
                slot.exp_avg_sq.len()
            );
        }
        if slot
            .exp_avg
            .iter()
            .chain(&slot.exp_avg_sq)
            .any(|value| !value.is_finite())
        {
            bail!(
                "AdamW optimizer slot {:?} contains non-finite moments",
                slot.name
            );
        }
        owned.push((
            format!("optimizer.{}.exp_avg", slot.name),
            vec![slot.exp_avg.len()],
            f32_bytes(&slot.exp_avg),
        ));
        owned.push((
            format!("optimizer.{}.exp_avg_sq", slot.name),
            vec![slot.exp_avg_sq.len()],
            f32_bytes(&slot.exp_avg_sq),
        ));
    }
    let mut views = Vec::with_capacity(owned.len());
    for (name, shape, bytes) in &owned {
        views.push((
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes)?,
        ));
    }
    let slot_names: Vec<&str> = state.slots.iter().map(|slot| slot.name.as_str()).collect();
    let slot_steps: Vec<u32> = state.slots.iter().map(|slot| slot.step).collect();
    let slot_decay_classes = state
        .slots
        .iter()
        .map(|slot| {
            slot.decay_class
                .with_context(|| {
                    format!(
                        "AdamW optimizer slot {:?} is missing its portable decay class",
                        slot.name
                    )
                })
                .map(AdamWDecayClass::checkpoint_label)
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = HashMap::from([
        (
            "format".to_string(),
            "hierarchos-vulkan-adamw-v3".to_string(),
        ),
        ("step".to_string(), state.step.to_string()),
        (
            "slot_names".to_string(),
            serde_json::to_string(&slot_names)?,
        ),
        (
            "slot_steps".to_string(),
            serde_json::to_string(&slot_steps)?,
        ),
        (
            "slot_decay_classes".to_string(),
            serde_json::to_string(&slot_decay_classes)?,
        ),
        ("layout".to_string(), "pytorch-row-major".to_string()),
    ]);
    serialize_to_file(views, Some(metadata), path)?;
    Ok(())
}

pub fn read_adamw_optimizer_state(path: &Path) -> Result<AdamWOptimizerState> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let (_, header) = SafeTensors::read_metadata(&bytes)?;
    let metadata = header
        .metadata()
        .as_ref()
        .context("AdamW optimizer checkpoint is missing metadata")?;
    let format = metadata.get("format").map(String::as_str);
    if !matches!(
        format,
        Some("hierarchos-vulkan-adamw-v1")
            | Some("hierarchos-vulkan-adamw-v2")
            | Some("hierarchos-vulkan-adamw-v3")
    ) {
        bail!("unsupported/missing AdamW optimizer checkpoint format metadata");
    }
    let step = metadata
        .get("step")
        .context("AdamW optimizer checkpoint is missing step metadata")?
        .parse::<u32>()
        .context("AdamW optimizer checkpoint step is not a u32")?;
    let names: Vec<String> = serde_json::from_str(
        metadata
            .get("slot_names")
            .context("AdamW optimizer checkpoint is missing slot_names metadata")?,
    )
    .context("decoding AdamW optimizer slot_names metadata")?;
    if names.is_empty() {
        bail!("AdamW optimizer checkpoint contains no slots");
    }
    let slot_steps = if matches!(
        format,
        Some("hierarchos-vulkan-adamw-v2") | Some("hierarchos-vulkan-adamw-v3")
    ) {
        let steps: Vec<u32> = serde_json::from_str(
            metadata
                .get("slot_steps")
                .context("AdamW v2 optimizer checkpoint is missing slot_steps metadata")?,
        )
        .context("decoding AdamW optimizer slot_steps metadata")?;
        if steps.len() != names.len() {
            bail!(
                "AdamW optimizer checkpoint has {} slot steps for {} slots",
                steps.len(),
                names.len()
            );
        }
        steps
    } else {
        // v1 was written before inactive fixed-registry slots existed, so every
        // registered tensor necessarily advanced on every global optimizer step.
        vec![step; names.len()]
    };
    let slot_decay_classes = if format == Some("hierarchos-vulkan-adamw-v3") {
        let labels: Vec<String> = serde_json::from_str(
            metadata
                .get("slot_decay_classes")
                .context("AdamW v3 optimizer checkpoint is missing slot_decay_classes metadata")?,
        )
        .context("decoding AdamW optimizer slot_decay_classes metadata")?;
        if labels.len() != names.len() {
            bail!(
                "AdamW optimizer checkpoint has {} decay classes for {} slots",
                labels.len(),
                names.len()
            );
        }
        labels
            .iter()
            .map(|label| AdamWDecayClass::from_checkpoint_label(label).map(Some))
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![None; names.len()]
    };

    let mut slots = Vec::with_capacity(names.len());
    for ((name, slot_step), decay_class) in
        names.into_iter().zip(slot_steps).zip(slot_decay_classes)
    {
        if slot_step > step {
            bail!("AdamW optimizer slot {name:?} step {slot_step} exceeds global step {step}");
        }
        let avg_name = format!("optimizer.{name}.exp_avg");
        let avg_sq_name = format!("optimizer.{name}.exp_avg_sq");
        let exp_avg = read_f32_tensor_from_set(&tensors, &avg_name)?;
        let exp_avg_sq = read_f32_tensor_from_set(&tensors, &avg_sq_name)?;
        if exp_avg.len() != exp_avg_sq.len() {
            bail!(
                "AdamW optimizer slot {name:?} moment lengths differ: {}/{}",
                exp_avg.len(),
                exp_avg_sq.len()
            );
        }
        slots.push(AdamWOptimizerSlotState {
            name,
            step: slot_step,
            decay_class,
            exp_avg,
            exp_avg_sq,
        });
    }
    Ok(AdamWOptimizerState { step, slots })
}

/// Write the in-flight full-model gradient registry using canonical model
/// parameter names. The file is intentionally ordinary F32 SafeTensors so a
/// PyTorch resume path can assign each tensor directly to `parameter.grad`,
/// while Vulkan can restore the same values into its device-resident registry.
pub fn write_pending_gradient_state(path: &Path, state: &[RwkvParameterSnapshot]) -> Result<()> {
    if state.is_empty() {
        bail!("pending gradient checkpoint requires at least one tensor");
    }
    let mut seen = std::collections::HashSet::new();
    let mut owned = Vec::with_capacity(state.len());
    for tensor in state {
        if tensor.name.trim().is_empty() || !seen.insert(tensor.name.as_str()) {
            bail!(
                "pending gradient checkpoint has empty/duplicate tensor {:?}",
                tensor.name
            );
        }
        if tensor.values.is_empty() {
            bail!(
                "pending gradient checkpoint tensor {:?} is empty",
                tensor.name
            );
        }
        if tensor.values.iter().any(|value| !value.is_finite()) {
            bail!(
                "pending gradient checkpoint tensor {:?} contains non-finite values",
                tensor.name
            );
        }
        owned.push((
            tensor.name.clone(),
            vec![tensor.values.len()],
            f32_bytes(&tensor.values),
        ));
    }
    let mut views = Vec::with_capacity(owned.len());
    for (name, shape, bytes) in &owned {
        views.push((
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes)?,
        ));
    }
    let names: Vec<&str> = state.iter().map(|tensor| tensor.name.as_str()).collect();
    let metadata = HashMap::from([
        (
            "format".to_string(),
            "hierarchos-vulkan-pending-gradients-v1".to_string(),
        ),
        ("slot_names".to_string(), serde_json::to_string(&names)?),
        ("layout".to_string(), "pytorch-row-major".to_string()),
    ]);
    serialize_to_file(views, Some(metadata), path)?;
    Ok(())
}

pub fn read_pending_gradient_state(path: &Path) -> Result<Vec<RwkvParameterSnapshot>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let (_, header) = SafeTensors::read_metadata(&bytes)?;
    let metadata = header
        .metadata()
        .as_ref()
        .context("pending gradient checkpoint is missing metadata")?;
    if metadata.get("format").map(String::as_str) != Some("hierarchos-vulkan-pending-gradients-v1")
    {
        bail!("unsupported/missing pending gradient checkpoint format metadata");
    }
    let names: Vec<String> = serde_json::from_str(
        metadata
            .get("slot_names")
            .context("pending gradient checkpoint is missing slot_names metadata")?,
    )
    .context("decoding pending gradient checkpoint slot_names metadata")?;
    if names.is_empty() {
        bail!("pending gradient checkpoint contains no tensors");
    }
    let mut seen = std::collections::HashSet::new();
    let mut state = Vec::with_capacity(names.len());
    for name in names {
        if name.trim().is_empty() || !seen.insert(name.clone()) {
            bail!(
                "pending gradient checkpoint has empty/duplicate tensor {:?}",
                name
            );
        }
        let values = read_f32_tensor_from_set(&tensors, &name)?;
        if values.is_empty() {
            bail!("pending gradient checkpoint tensor {name:?} is empty");
        }
        state.push(RwkvParameterSnapshot { name, values });
    }
    Ok(state)
}

fn read_f32_tensor_from_set(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing optimizer tensor {name:?}"))?;
    if tensor.dtype() != Dtype::F32 || tensor.shape().len() != 1 {
        bail!(
            "optimizer tensor {name:?} must be rank-1 F32, got {:?} {:?}",
            tensor.dtype(),
            tensor.shape()
        );
    }
    let raw = tensor.data();
    if raw.len() % 4 != 0 {
        bail!(
            "optimizer tensor {name:?} has invalid FP32 byte length {}",
            raw.len()
        );
    }
    let mut values = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("optimizer tensor {name:?} contains non-finite values");
    }
    Ok(values)
}

fn validate_f32_payload(name: &str, shape: &[usize], values: &[f32]) -> Result<()> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .context("tensor element count overflow")?;
    if expected != values.len() {
        bail!(
            "tensor {name:?} shape {shape:?} expects {expected} values, got {}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("tensor {name:?} contains non-finite values");
    }
    Ok(())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn is_supported_model_float_dtype(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let encoded = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut mantissa = u32::from(fraction);
            let mut exponent32 = 113u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent32 -= 1;
            }
            mantissa &= 0x03ff;
            sign | (exponent32 << 23) | (mantissa << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (u32::from(fraction) << 13),
    };
    f32::from_bits(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_optimizer_path(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hierarchos-vulkan-{label}-{}-{nonce}.safetensors",
            std::process::id()
        ))
    }

    fn u16_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn portable_running_carrier_reader_decodes_python_cached_rosa_layout() -> Result<()> {
        let tensor_path = temp_optimizer_path("portable-running-carriers");
        let replay_path = tensor_path.with_extension("json");
        write_portable_replay_tensors(
            &tensor_path,
            PORTABLE_TRAINING_REPLAY_FORMAT,
            &[
                HierarchosPortableReplayTensor::f32(
                    "h",
                    vec![2, 1, 3],
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ),
                HierarchosPortableReplayTensor::f32(
                    "l",
                    vec![2, 1, 3],
                    vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ),
                HierarchosPortableReplayTensor::f32("prev", vec![2, 2], vec![0.25, 0.5, 0.75, 1.0]),
                HierarchosPortableReplayTensor::f32(
                    "target",
                    vec![2, 2],
                    vec![1.25, 1.5, 1.75, 2.0],
                ),
                HierarchosPortableReplayTensor::f32(
                    "drift",
                    vec![2, 2],
                    vec![-0.25, -0.5, 0.25, 0.5],
                ),
                HierarchosPortableReplayTensor::u8(
                    "past_tokens",
                    vec![2, 3],
                    vec![2, 5, 2, 3, 4, 3],
                ),
            ],
        )?;
        let replay = serde_json::json!({
            "format": PORTABLE_TRAINING_REPLAY_FORMAT,
            "state": {
                "__kind__": "dict",
                "items": [[
                    "running_states",
                    {
                        "__kind__": "tuple",
                        "items": [
                            {"__kind__": "tensor", "name": "h"},
                            {"__kind__": "tensor", "name": "l"},
                            {"__kind__": "tensor", "name": "prev"},
                            {"__kind__": "tensor", "name": "target"},
                            {"__kind__": "tensor", "name": "drift"},
                            {
                                "__kind__": "tuple",
                                "items": [
                                    null,
                                    null,
                                    {"__kind__": "tensor", "name": "past_tokens"},
                                    null,
                                    null,
                                    null,
                                    null
                                ]
                            }
                        ]
                    }
                ]]
            }
        });
        fs::write(&replay_path, serde_json::to_vec(&replay)?)?;

        let carriers = read_portable_running_carriers(&replay_path, Some(&tensor_path))?
            .context("portable replay should contain running carriers")?;
        let _ = fs::remove_file(&replay_path);
        let _ = fs::remove_file(&tensor_path);

        assert_eq!(carriers.h_state.shape, vec![2, 1, 3]);
        assert_eq!(carriers.h_state.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(carriers.l_state.shape, vec![2, 1, 3]);
        assert_eq!(carriers.previous_context.values, vec![0.25, 0.5, 0.75, 1.0]);
        assert_eq!(carriers.target_context.values, vec![1.25, 1.5, 1.75, 2.0]);
        assert_eq!(carriers.drift_state.values, vec![-0.25, -0.5, 0.25, 0.5]);
        assert_eq!(
            carriers.rosa_token_histories,
            vec![vec![2, 5, 2], vec![3, 4, 3]]
        );
        Ok(())
    }

    #[test]
    fn portable_running_carrier_encoder_roundtrips_python_abi() -> Result<()> {
        let tensor_path = temp_optimizer_path("portable-running-roundtrip");
        let replay_path = tensor_path.with_extension("json");
        let carriers = HierarchosPortableRunningCarriers {
            h_state: HierarchosPortableReplayFloatTensor {
                shape: vec![1, 1, 2],
                values: vec![1.0, 2.0],
            },
            l_state: HierarchosPortableReplayFloatTensor {
                shape: vec![1, 1, 2],
                values: vec![3.0, 4.0],
            },
            previous_context: HierarchosPortableReplayFloatTensor {
                shape: vec![1, 2],
                values: vec![0.25, 0.5],
            },
            target_context: HierarchosPortableReplayFloatTensor {
                shape: vec![1, 2],
                values: vec![0.75, 1.0],
            },
            drift_state: HierarchosPortableReplayFloatTensor {
                shape: vec![1, 2],
                values: vec![0.0, 0.0],
            },
            rosa_token_histories: vec![vec![7, 3, 7, 9]],
            ltm_state: HierarchosPortableLtmRunningState {
                fast_vals: Some(HierarchosPortableReplayFloatTensor {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                }),
                mom_vals: Some(HierarchosPortableReplayFloatTensor {
                    shape: vec![2, 2],
                    values: vec![0.1, 0.2, 0.3, 0.4],
                }),
                timestamps: Some(HierarchosPortableReplayFloatTensor {
                    shape: vec![2],
                    values: vec![5.0, 6.0],
                }),
                sources: Some(HierarchosPortableReplayI64Tensor {
                    shape: vec![2],
                    values: vec![2, 4],
                }),
                wallclock_timestamps: Some(HierarchosPortableReplayF64Tensor {
                    shape: vec![2],
                    values: vec![10.0, 11.0],
                }),
            },
        };
        carriers.ltm_state.validate_exact_shared_geometry(2, 2)?;
        let (encoded, tensors) = encode_portable_running_carriers(&carriers)?;
        write_portable_replay_tensors(&tensor_path, PORTABLE_TRAINING_REPLAY_FORMAT, &tensors)?;
        fs::write(
            &replay_path,
            serde_json::to_vec(&serde_json::json!({
                "format": PORTABLE_TRAINING_REPLAY_FORMAT,
                "state": {
                    "__kind__": "dict",
                    "items": [["running_states", encoded]]
                }
            }))?,
        )?;

        let decoded = read_portable_running_carriers(&replay_path, Some(&tensor_path))?
            .context("roundtrip replay should contain running carriers")?;
        let _ = fs::remove_file(&replay_path);
        let _ = fs::remove_file(&tensor_path);
        assert_eq!(decoded, carriers);
        Ok(())
    }

    #[test]
    fn portable_ltm_geometry_rejects_batch_isolated_pytorch_state() {
        let state = HierarchosPortableLtmRunningState {
            fast_vals: Some(HierarchosPortableReplayFloatTensor {
                shape: vec![2, 3, 4],
                values: vec![0.0; 24],
            }),
            mom_vals: Some(HierarchosPortableReplayFloatTensor {
                shape: vec![2, 3, 4],
                values: vec![0.0; 24],
            }),
            timestamps: Some(HierarchosPortableReplayFloatTensor {
                shape: vec![2, 3],
                values: vec![0.0; 6],
            }),
            sources: None,
            wallclock_timestamps: None,
        };
        let error = state
            .validate_exact_shared_geometry(3, 4)
            .expect_err("batch-isolated writable LTM state must not be flattened");
        assert!(error.to_string().contains("native shared LTM geometry"));
    }

    #[test]
    fn model_tensor_reader_promotes_f16_and_bf16_to_fp32() -> Result<()> {
        let path = temp_optimizer_path("mixed-model-read");
        let fp16 = u16_bytes(&[0x3c00, 0xc000, 0x0001]);
        let bf16 = u16_bytes(&[0x3f80, 0xc000, 0x3f00]);
        let views = vec![
            ("fp16.weight", TensorView::new(Dtype::F16, vec![3], &fp16)?),
            ("bf16.weight", TensorView::new(Dtype::BF16, vec![3], &bf16)?),
        ];
        serialize_to_file(views, None, &path)?;

        let (_, fp16_values) = read_f32_tensor(&path, "fp16.weight")?;
        let (_, bf16_values) = read_f32_tensor(&path, "bf16.weight")?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(fp16_values, vec![1.0, -2.0, 2f32.powi(-24)]);
        assert_eq!(bf16_values, vec![1.0, -2.0, 0.5]);
        Ok(())
    }

    #[test]
    fn fp32_replacement_promotes_trainable_mixed_source_and_preserves_other_dtype() -> Result<()> {
        let source = temp_optimizer_path("mixed-model-source");
        let destination = temp_optimizer_path("mixed-model-destination");
        let lm_head = u16_bytes(&[0x3c00, 0xc000]);
        let untouched = u16_bytes(&[0x3f00]);
        let views = vec![
            (
                "lm_head.weight",
                TensorView::new(Dtype::F16, vec![1, 2], &lm_head)?,
            ),
            (
                "untouched.weight",
                TensorView::new(Dtype::BF16, vec![1], &untouched)?,
            ),
        ];
        serialize_to_file(views, None, &source)?;

        replace_f32_tensor(
            &source,
            &destination,
            "lm_head.weight",
            &[1, 2],
            &[1.25, -0.75],
        )?;
        let destination_bytes = fs::read(&destination)?;
        let tensors = SafeTensors::deserialize(&destination_bytes)?;
        assert_eq!(tensors.tensor("lm_head.weight")?.dtype(), Dtype::F32);
        assert_eq!(tensors.tensor("untouched.weight")?.dtype(), Dtype::BF16);
        drop(tensors);
        let (_, values) = read_f32_tensor(&destination, "untouched.weight")?;
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&destination);
        assert_eq!(values, vec![0.5]);
        Ok(())
    }

    #[test]
    fn portable_master_validator_rejects_low_precision_optimizer_slot() -> Result<()> {
        let path = temp_optimizer_path("portable-master-dtype");
        let fp32 = f32_bytes(&[1.0, -2.0]);
        let fp16 = u16_bytes(&[0x3c00, 0xc000]);
        let views = vec![
            (
                "master.weight",
                TensorView::new(Dtype::F32, vec![2], &fp32)?,
            ),
            (
                "mirror.weight",
                TensorView::new(Dtype::F16, vec![2], &fp16)?,
            ),
        ];
        serialize_to_file(views, None, &path)?;

        validate_f32_tensor_names(&path, &["master.weight"])?;
        let error = validate_f32_tensor_names(&path, &["mirror.weight"])
            .expect_err("F16 optimizer-bound master must be rejected");
        let _ = std::fs::remove_file(&path);
        let message = error.to_string();
        assert!(message.contains("mirror.weight"), "{message}");
        assert!(message.contains("F16"), "{message}");
        Ok(())
    }

    #[test]
    fn adamw_v3_roundtrip_preserves_slot_steps_and_decay_topology() -> Result<()> {
        let path = temp_optimizer_path("adamw-v3-slot-semantics");
        let state = AdamWOptimizerState {
            step: 7,
            slots: vec![
                AdamWOptimizerSlotState {
                    name: "always.weight".to_string(),
                    step: 7,
                    decay_class: Some(AdamWDecayClass::Decay),
                    exp_avg: vec![0.1, -0.2],
                    exp_avg_sq: vec![0.01, 0.04],
                },
                AdamWOptimizerSlotState {
                    name: "val_proj.weight".to_string(),
                    step: 3,
                    decay_class: Some(AdamWDecayClass::NoDecay),
                    exp_avg: vec![0.3, -0.4],
                    exp_avg_sq: vec![0.09, 0.16],
                },
            ],
        };
        write_adamw_optimizer_state(&path, &state)?;
        let restored = read_adamw_optimizer_state(&path)?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.step, 7);
        assert_eq!(restored.slots.len(), 2);
        assert_eq!(restored.slots[0].step, 7);
        assert_eq!(restored.slots[1].step, 3);
        assert_eq!(restored.slots[0].decay_class, Some(AdamWDecayClass::Decay));
        assert_eq!(
            restored.slots[1].decay_class,
            Some(AdamWDecayClass::NoDecay)
        );
        assert_eq!(restored.slots[1].exp_avg, vec![0.3, -0.4]);
        assert_eq!(restored.slots[1].exp_avg_sq, vec![0.09, 0.16]);
        Ok(())
    }

    #[test]
    fn adamw_v1_reader_maps_historical_global_step_to_every_slot() -> Result<()> {
        let path = temp_optimizer_path("adamw-v1-backcompat");
        let avg = f32_bytes(&[0.1, 0.2]);
        let avg_sq = f32_bytes(&[0.01, 0.04]);
        let views = vec![
            (
                "optimizer.val_proj.weight.exp_avg",
                TensorView::new(Dtype::F32, vec![2], &avg)?,
            ),
            (
                "optimizer.val_proj.weight.exp_avg_sq",
                TensorView::new(Dtype::F32, vec![2], &avg_sq)?,
            ),
        ];
        let metadata = HashMap::from([
            (
                "format".to_string(),
                "hierarchos-vulkan-adamw-v1".to_string(),
            ),
            ("step".to_string(), "5".to_string()),
            (
                "slot_names".to_string(),
                serde_json::to_string(&["val_proj.weight"])?,
            ),
            ("layout".to_string(), "pytorch-row-major".to_string()),
        ]);
        serialize_to_file(views, Some(metadata), &path)?;
        let restored = read_adamw_optimizer_state(&path)?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.step, 5);
        assert_eq!(restored.slots[0].step, 5);
        assert_eq!(restored.slots[0].decay_class, None);
        Ok(())
    }

    #[test]
    fn adamw_v2_reader_preserves_slot_steps_with_unknown_legacy_decay_topology() -> Result<()> {
        let path = temp_optimizer_path("adamw-v2-backcompat");
        let avg = f32_bytes(&[0.1, 0.2]);
        let avg_sq = f32_bytes(&[0.01, 0.04]);
        let views = vec![
            (
                "optimizer.val_proj.weight.exp_avg",
                TensorView::new(Dtype::F32, vec![2], &avg)?,
            ),
            (
                "optimizer.val_proj.weight.exp_avg_sq",
                TensorView::new(Dtype::F32, vec![2], &avg_sq)?,
            ),
        ];
        let metadata = HashMap::from([
            (
                "format".to_string(),
                "hierarchos-vulkan-adamw-v2".to_string(),
            ),
            ("step".to_string(), "5".to_string()),
            (
                "slot_names".to_string(),
                serde_json::to_string(&["val_proj.weight"])?,
            ),
            ("slot_steps".to_string(), serde_json::to_string(&[3u32])?),
            ("layout".to_string(), "pytorch-row-major".to_string()),
        ]);
        serialize_to_file(views, Some(metadata), &path)?;
        let restored = read_adamw_optimizer_state(&path)?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.step, 5);
        assert_eq!(restored.slots[0].step, 3);
        assert_eq!(restored.slots[0].decay_class, None);
        Ok(())
    }

    #[test]
    fn pending_gradient_roundtrip_preserves_registry_order_and_values() -> Result<()> {
        let path = temp_optimizer_path("pending-gradients-v1");
        let state = vec![
            RwkvParameterSnapshot {
                name: "lm_head.weight".to_string(),
                values: vec![0.5, -0.25, 0.125],
            },
            RwkvParameterSnapshot {
                name: "h_rnn.time_mix_k".to_string(),
                values: vec![1.0, -2.0],
            },
        ];
        write_pending_gradient_state(&path, &state)?;
        let restored = read_pending_gradient_state(&path)?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].name, "lm_head.weight");
        assert_eq!(restored[0].values, vec![0.5, -0.25, 0.125]);
        assert_eq!(restored[1].name, "h_rnn.time_mix_k");
        assert_eq!(restored[1].values, vec![1.0, -2.0]);
        Ok(())
    }
}
