//! Vulkan-first training primitives for Hierarchos and decoder-only Transformers.
//!
//! The crate keeps the original coherent-v9/RWKV Hierarchos graph while also
//! exposing a native [`VulkanTransformer`] graph for Hugging Face causal-LM
//! packages. Both paths share the same Vulkan device/runtime, optimizer
//! primitives, checkpoint conventions, and FP32-master training foundation.

mod adapter;
mod checkpoint;
mod control;
mod full_training_graph;
mod lm_execution;
mod lora;
mod mixed_precision;
mod projection;
mod projection_graph;
mod rwkv_adapter_channel_mix;
mod rwkv_cell;
mod rwkv_channel_mix;
mod rwkv_low_rank;
mod rwkv_optimizer;
mod rwkv_packed_state;
mod rwkv_post_mix;
mod rwkv_state;
mod rwkv_tbptt;
mod rwkv_time_mix_core;
mod shared_lm_head;
mod stochastic;
mod tape_profiles;
mod tied_embedding;
mod token_frontend;
mod training_numerics;
mod transformer;
mod vulkan;

use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use hierarchos_inference::ModelConfig;

pub use adapter::{AdapterStepResult, SharedTokenAdapterTrainer};
pub use checkpoint::{
    decode_portable_replay_json, encode_portable_replay_json, encode_portable_running_carriers,
    read_adamw_optimizer_state, read_f32_tensor, read_model_ltm_running_state,
    read_pending_gradient_state, read_portable_replay_json_field, read_portable_running_carriers,
    replace_f32_tensor, replace_f32_tensor_values, replace_f32_tensors, validate_f32_tensor_names,
    write_adamw_optimizer_state, write_f32_tensor, write_pending_gradient_state,
    write_portable_replay_tensors, HierarchosPortableLtmRunningState,
    HierarchosPortableReplayFloatTensor, HierarchosPortableReplayTensor,
    HierarchosPortableReplayTensorData, HierarchosPortableRunningCarriers,
};
pub use control::{
    ContextDriftVulkanOp, ContextLerpConcatResult, DriftUpdateResult, HardActResult,
    HardActVulkanOp,
};
pub use full_training_graph::{
    hierarchos_portable_fisher_yates_permutation, HierarchosBudgetedTokenTapeTrainResult,
    HierarchosDataStreamCursorState, HierarchosDeviceResidentDynamicWavefrontResult,
    HierarchosDynamicGradientClipStepResult, HierarchosDynamicLossScaleStepResult,
    HierarchosExecutionMirrorContract, HierarchosExecutionPolicyState,
    HierarchosFullModelReplicaState, HierarchosFullModelReplicaTransportSource,
    HierarchosFullModelUpdateMode, HierarchosFusedRecurrentLossStepResult,
    HierarchosGradientClipStepResult, HierarchosGradientStreamStats,
    HierarchosJointTrainingAdapterConstraint, HierarchosJointTrainingAdapterMemoryPlan,
    HierarchosLabeledSequenceObjective, HierarchosLearningRateScheduleState,
    HierarchosLossCoupledTokenInput, HierarchosLossCoupledTokenStepResult,
    HierarchosLossScaleStepDecision, HierarchosLossScalingState, HierarchosLowRankG2GradientTrace,
    HierarchosLtmAlignmentControllerState, HierarchosManagerHardActInput,
    HierarchosManagerHardActResult, HierarchosParameterStateContract,
    HierarchosPendingGradientShard, HierarchosPendingGradientTransportSource,
    HierarchosPortableTrainingReplay, HierarchosProjectionCoupledTokenInput,
    HierarchosProjectionCoupledTokenStepResult, HierarchosRawTokenLabeledSequenceInput,
    HierarchosRawTokenSequenceContextInput, HierarchosRawTokenTapeMicrobatchInput,
    HierarchosRawTokenTapeStepInput, HierarchosRawTokenWorkerRefinementLossInput,
    HierarchosRecurrentBranchInput, HierarchosReplicaStateDeviceGroupTransport,
    HierarchosReplicaStateRangeRetirement, HierarchosReplicaStateRetirementTimeline,
    HierarchosReplicaStateStreamStats, HierarchosReplicaStateTimelineReservation,
    HierarchosSequenceGradientNormalization, HierarchosSequenceStateArena,
    HierarchosSequenceStateSnapshot, HierarchosStochasticRngPolicyState,
    HierarchosTapeMemoryPolicy, HierarchosTokenTape, HierarchosTokenTapeArena,
    HierarchosTokenTapeControlSnapshot, HierarchosTokenTapeFootprint,
    HierarchosTokenTapeMemoryPlan, HierarchosTokenTapeMicrobatchInput,
    HierarchosTokenTapeMicrobatchTrainResult, HierarchosTokenTapeReadbackPolicy,
    HierarchosTokenTapeStepInput, HierarchosTokenTapeTrainResult, HierarchosTokenTapeUpdateMode,
    HierarchosTrainingCheckpointManifest, HierarchosTrainingGraph, HierarchosTrainingGraphSummary,
    HierarchosTrainingPrecisionPolicy, HierarchosTrainingSessionState,
    HierarchosTrainingWorkingSetEntry, HierarchosTrainingWorkingSetEpoch,
    HierarchosTrainingWorkingSetLifetime, HierarchosTrainingWorkingSetPlan,
    HierarchosTrainingWorkingSetSlot, HierarchosWorkerRefinementLossInput,
    HierarchosWorkerRefinementLossStepResult, HIERARCHOS_VULKAN_DATA_STREAM_CURSOR_FORMAT,
    HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT, HIERARCHOS_VULKAN_GRADIENTS_FILENAME,
    HIERARCHOS_VULKAN_LM_FORCE_CANONICAL_GRAD_STAGING_ENV,
    HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_COMPENSATED_ENV,
    HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_WIDEN_PRODUCT_ENV,
    HIERARCHOS_VULKAN_OPTIMIZER_FILENAME, HIERARCHOS_VULKAN_PARAMETER_STATE_FORMAT,
    HIERARCHOS_VULKAN_PORTABLE_REPLAY_FILENAME, HIERARCHOS_VULKAN_PORTABLE_REPLAY_FORMAT,
    HIERARCHOS_VULKAN_PORTABLE_REPLAY_TENSOR_FILENAME,
    HIERARCHOS_VULKAN_PORTABLE_SAMPLER_RNG_ALGORITHM, HIERARCHOS_VULKAN_TRAINING_MANIFEST_FILENAME,
    HIERARCHOS_VULKAN_TRAINING_PRECISION_ENV, HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT,
};
pub use lm_execution::{
    HierarchosLmExecutionArm, HierarchosLmWeightGradTopology,
    HIERARCHOS_VULKAN_LM_AUTOTUNE_CACHE_PATH_ENV, HIERARCHOS_VULKAN_LM_AUTOTUNE_DISABLE_ENV,
    HIERARCHOS_VULKAN_LM_AUTOTUNE_LOG_ENV, HIERARCHOS_VULKAN_LM_BACKWARD_TOPOLOGY_ENV,
    HIERARCHOS_VULKAN_LM_DISABLE_PERSISTENT_CACHE_ENV, HIERARCHOS_VULKAN_LM_EXECUTION_ARM_ENV,
    HIERARCHOS_VULKAN_LM_REAUTOTUNE_ENV,
};
pub use lora::{
    merge_hierarchos_lora_safetensors, HierarchosNativeLoraMergeReport,
    HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME, HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME,
    HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME,
};
pub use mixed_precision::{VulkanFp32MasterParameterMirror, VulkanParameterStorageFormat};
pub use projection::{LinearProjectionTrainer, ProjectionStepResult};
pub use rwkv_adapter_channel_mix::{RwkvAdapterChannelMixOp, RwkvAdapterChannelMixResult};
pub use rwkv_cell::{RwkvCellSliceOp, RwkvCellSliceResult, RwkvPackedCellOp, RwkvPackedCellResult};
pub use rwkv_channel_mix::{RwkvChannelMixOp, RwkvChannelMixResult};
pub use rwkv_low_rank::{
    RwkvLowRankOp, RwkvLowRankParameterGradArithmetic, RwkvLowRankResult,
    RwkvLowRankWeightGradBenchmark,
};
pub use rwkv_optimizer::{
    AdamWDecayClass, AdamWOptimizerSlotState, AdamWOptimizerState, RwkvOptimizerStepResult,
    RwkvParameterSnapshot,
};
pub use rwkv_packed_state::{RwkvPackedStateOp, RwkvPackedStateResult, RwkvStateReadoutMode};
pub use rwkv_post_mix::{RwkvPostMixOp, RwkvPostMixResult};
pub use rwkv_state::{RwkvMatrixStateOp, RwkvMatrixStateResult};
pub use rwkv_tbptt::{
    RwkvRecurrentGradientMode, RwkvTbpttBranchInput, RwkvTbpttSchedule, RwkvTbpttSequenceOp,
    RwkvTbpttSequenceResult, RwkvTbpttTrainStepResult, SharedLmHeadTrainMode,
};
pub use rwkv_time_mix_core::{
    RwkvFullTimeMixResult, RwkvFusedTimeMixCoreResult, RwkvNumericsPolicy, RwkvTimeMixCoreOp,
    RwkvTimeMixCoreResult,
};
pub use shared_lm_head::SharedLmHeadParameter;
pub use stochastic::{
    hierarchos_canonical_philox_word, hierarchos_dropout_threshold, CanonicalDropoutVulkanOp,
    HierarchosCanonicalRngReservation, HierarchosCanonicalRngState,
    HIERARCHOS_CANONICAL_COUNTER_RNG_ALGORITHM,
};
pub use tape_profiles::{
    HierarchosTokenTapeProfileDatabase, HierarchosTokenTapeProfileScore,
    BACKWARD_KERNEL_GEOMETRY_POLICY_REVISION, HIERARCHOS_TOKEN_TAPE_EXPLORE_EVERY_ENV,
    HIERARCHOS_TOKEN_TAPE_ONLINE_AUTOTUNE_DISABLE_ENV, HIERARCHOS_TOKEN_TAPE_PROFILE_DISABLE_ENV,
    HIERARCHOS_TOKEN_TAPE_PROFILE_ENV, HIERARCHOS_TOKEN_TAPE_PROFILE_FILENAME,
    HIERARCHOS_TOKEN_TAPE_PROFILE_LOG_ENV,
};
pub use tied_embedding::{TiedTokenEmbeddingOp, TiedTokenEmbeddingResult};
pub use token_frontend::{
    HierarchosLtmValueAlignmentResult, HierarchosTokenFrontendBackwardResult,
    HierarchosTokenFrontendForwardResult, HierarchosTokenFrontendInput, HierarchosTokenFrontendOp,
    HierarchosTokenMemoryFrontendBackwardResult, HierarchosTokenMemoryFrontendForwardResult,
    HierarchosTokenMemoryFrontendInput, HierarchosTokenMemoryFrontendLaneInput,
};
pub use training_numerics::VulkanGradientNonfiniteDetector;
pub use transformer::{
    VulkanSeq2SeqTransformer, VulkanTransformer, VulkanTransformerArchitecture,
    VulkanTransformerConfig, VulkanTransformerGenerationAttention,
    VulkanTransformerGenerationConfig, VulkanTransformerGenerationHiddenState,
    VulkanTransformerGenerationMode, VulkanTransformerGenerationOutput,
    VulkanTransformerGenerationSequenceOutput, VulkanTransformerLoraConfig,
    VulkanTransformerStepResult, VULKAN_TRANSFORMER_ADAPTER_CONFIG_FILENAME,
    VULKAN_TRANSFORMER_ADAPTER_WEIGHTS_FILENAME,
};
pub use vulkan::{
    GpuBuffer, VulkanDevice, VulkanDeviceGroupInfo, VulkanExternalBufferCapabilities,
    VulkanGradientTransportBackend, VulkanGradientTransportPlan, VulkanMemoryBudget,
    VulkanMemoryHeapBudget, VulkanMemoryStats, VulkanMixedPrecisionCapabilities,
    VulkanPhysicalDeviceInfo, VulkanSubmissionArenaStats,
};

pub const HIERARCHOS_VULKAN_DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV: &str =
    "HIERARCHOS_VULKAN_DISABLE_NATIVE_FP16_LM_INPUT_GRAD";
/// Profiling escape hatch for the pre-direct-dW full-graph path. When set, the
/// dense LM gradient is staged through the dedicated full-matrix scratch buffer
/// and then accumulated into the freshly-zeroed tied gradient, reproducing the
/// old bandwidth boundary for same-binary A/B measurements.
pub const HIERARCHOS_VULKAN_LM_FORCE_DENSE_GRAD_STAGING_ENV: &str =
    "HIERARCHOS_VULKAN_LM_FORCE_DENSE_GRAD_STAGING";

const LINEAR_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_forward.spv");
#[cfg(test)]
const LAYER_NORM_LINEAR_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_linear_forward_fused.spv");
const CROSS_ENTROPY_ROW_STATS_SPV: &[u8] = include_bytes!("../shaders/cross_entropy_row_stats.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad.spv");
#[cfg(test)]
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad.spv");
const CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_row_stats_streaming.spv");
const CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_row_stats_streaming_fp16_packed.spv");
const CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_TAPE_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_row_stats_streaming_fp16_packed_tape.spv");
const CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS8_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_logit_tape_fp16_packed_rows8.spv");
const CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_logit_tape_fp16_packed_rows16.spv");
const CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_DOT4_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_logit_tape_fp16_packed_rows16_dot4.spv");
const CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_CLUSTER4_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_logit_tape_fp16_packed_rows16_cluster4.spv");
const CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_row_stats_tile_partials.spv");
const CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_ROWS16_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_row_stats_tile_partials_rows16.spv");
#[cfg(test)]
const CROSS_ENTROPY_LOGITS_TO_GRAD_INPLACE_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_logits_to_grad_inplace.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_streaming.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS4_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed_rows4.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS16_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed_rows16.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS4_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_tape_fp16_packed_rows4.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_tape_fp16_packed.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS16_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_tape_fp16_packed_rows16.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_VOCAB_TILE: usize = 4;
const CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_VOCAB_TILE: usize = 64;
const CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_MAX_ROWS: usize = 8;
const CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_SHARED_BYTES: u32 = 16_384;
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_streaming.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_streaming_fp16_packed.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_TAPE_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_tape_fp16_packed.spv");
const CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.spv");
const CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_SPV: &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden.spv"
);
#[cfg(test)]
const CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE128_SPV: &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile128.spv"
);
#[cfg(test)]
const CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE128_WG128_SPV:
    &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile128_wg128.spv"
);
#[cfg(test)]
const CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_SPV: &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256.spv"
);
#[cfg(test)]
const CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_WG128_SPV:
    &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256_wg128.spv"
);
const CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_WG256_SPV:
    &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256_wg256.spv"
);
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_tile_reduce.spv");
#[cfg(test)]
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE128_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_tile_reduce_tile128.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE256_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_tile_reduce_tile256.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_streaming_fp16_native.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE64_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_streaming_fp16_native_reuse64.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE128_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_streaming_fp16_native_reuse128.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE224_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_streaming_fp16_native_reuse224.spv");
const CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_input_grad_streaming_fp16_native_compute.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_ROWS4_SPV: &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows4.spv"
);
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_linear_weight_grad_streaming_fp16_native_compute.spv");
const CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_ROWS16_SPV: &[u8] = include_bytes!(
    "../shaders/cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows16.spv"
);
const CROSS_ENTROPY_ROW_LOSS_EXTRACT_SPV: &[u8] =
    include_bytes!("../shaders/cross_entropy_row_loss_extract.spv");
#[cfg(test)]
const LINEAR_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_weight_grad.spv");
#[cfg(test)]
const LINEAR_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
#[cfg(test)]
const LAYER_NORM_FORWARD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_forward.spv");
const LAYER_NORM_STATS_SPV: &[u8] = include_bytes!("../shaders/layer_norm_stats.spv");
const LAYER_NORM_AFFINE_CLAMP_BACKWARD_INPLACE_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_affine_clamp_backward_inplace.spv");
const LAYER_NORM_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_input_grad.spv");
const LAYER_NORM_INPUT_GRAD_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_input_grad_fp16_native_compute.spv");
const LAYER_NORM_PARAM_GRAD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_param_grad.spv");
const LAYER_NORM_PARAM_GRAD_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_param_grad_fp16_native_compute.spv");
const EMBEDDING_TOKEN_SORT_SPV: &[u8] = include_bytes!("../shaders/embedding_token_sort.spv");
const EMBEDDING_GRAD_SEGMENTED_SPV: &[u8] =
    include_bytes!("../shaders/embedding_grad_segmented.spv");
const EMBEDDING_RADIX_HISTOGRAM_SPV: &[u8] =
    include_bytes!("../shaders/embedding_radix_histogram.spv");
const EMBEDDING_RADIX_PREFIX_SPV: &[u8] = include_bytes!("../shaders/embedding_radix_prefix.spv");
const EMBEDDING_RADIX_SCATTER_SPV: &[u8] = include_bytes!("../shaders/embedding_radix_scatter.spv");
const EMBEDDING_SEGMENTED_SORT_CAPACITY: usize = 1024;
const EMBEDDING_RADIX_BLOCK_SIZE: usize = 256;
const EMBEDDING_RADIX_BUCKETS: usize = 16;
const ADAMW_SPV: &[u8] = include_bytes!("../shaders/adamw.spv");

fn lm_execution_autotune_kernel_signature() -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for shader in [
        CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_SPV,
        CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_TAPE_SPV,
        CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS8_SPV,
        CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_SPV,
        CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_DOT4_SPV,
        CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_CLUSTER4_SPV,
        CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_SPV,
        CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_ROWS16_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_PACKED_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_TAPE_FP16_PACKED_SPV,
        CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_SPV,
        CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_SPV,
        CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_WG256_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE256_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE64_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE128_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE224_SPV,
        CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS4_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS16_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_ROWS4_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_ROWS16_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS4_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_SPV,
        CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS16_SPV,
    ] {
        for &byte in shader {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn create_fp16_lm_weight_grad_kernel(
    device: &VulkanDevice,
    topology: HierarchosLmWeightGradTopology,
    shader: &[u8],
) -> Result<Option<vulkan::ComputeKernel>> {
    if !device.supports_compute_work_group_size(topology.local_size()) {
        return Ok(None);
    }
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        shader,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadWrite,
        ],
        std::mem::size_of::<LmWeightGradPush>() as u32,
    )?))
}

fn create_native_fp16_lm_weight_grad_kernel(
    device: &VulkanDevice,
    topology: HierarchosLmWeightGradTopology,
    shader: &[u8],
) -> Result<Option<vulkan::ComputeKernel>> {
    if !device
        .mixed_precision_capabilities()
        .native_fp16_storage_compute_ready()
    {
        return Ok(None);
    }
    create_fp16_lm_weight_grad_kernel(device, topology, shader)
}

fn create_fp16_lm_tape_weight_grad_kernel(
    device: &VulkanDevice,
    topology: HierarchosLmWeightGradTopology,
    shader: &[u8],
) -> Result<Option<vulkan::ComputeKernel>> {
    if !device.supports_compute_work_group_size(topology.local_size()) {
        return Ok(None);
    }
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        shader,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadWrite,
        ],
        std::mem::size_of::<LmWeightGradPush>() as u32,
    )?))
}

fn create_native_fp16_lm_input_grad_kernel(
    device: &VulkanDevice,
    arm: HierarchosLmExecutionArm,
    shader: &[u8],
) -> Result<Option<vulkan::ComputeKernel>> {
    if !device
        .mixed_precision_capabilities()
        .native_fp16_storage_compute_ready()
    {
        return Ok(None);
    }
    let required_shared = arm
        .native_fp16_shared_memory_bytes()
        .with_context(|| format!("{} is not a native-FP16 LM input-adjoint arm", arm.label()))?;
    if required_shared > device.max_compute_shared_memory_bytes() {
        return Ok(None);
    }
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        shader,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
        ],
        std::mem::size_of::<LinearPush>() as u32,
    )?))
}

#[derive(Clone, Copy, Debug)]
pub struct AdamWHyperParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWHyperParams {
    fn default() -> Self {
        Self {
            lr: 1.0e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.1,
        }
    }
}

impl AdamWHyperParams {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            bail!("learning rate must be finite and non-negative");
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            bail!("AdamW betas must be in [0, 1)");
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            bail!("AdamW epsilon must be finite and non-negative");
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            bail!("AdamW weight decay must be finite and non-negative");
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct TrainStepResult {
    pub loss: f32,
    pub step: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
    z_loss_weight: f32,
    activation_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LmWeightGradPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
    accumulate: u32,
    z_loss_weight: f32,
    activation_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormAffineClampBackwardPush {
    rows: u32,
    dim: u32,
    max_abs: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CrossEntropyPush {
    rows: u32,
    vocab_size: u32,
    z_loss_weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WeightGradPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormForwardPush {
    rows: u32,
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormLinearForwardPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormBackwardPush {
    rows: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbeddingGradPush {
    token_count: u32,
    dim: u32,
    vocab_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbeddingRadixPush {
    token_count: u32,
    shift: u32,
    block_count: u32,
    source_is_identity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWPush {
    len: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

/// First trainable Vulkan slice of coherent-v9: the tied token embedding / LM
/// head with exact PyTorch `[vocab, context]` storage.
///
/// All forward, loss-gradient, parameter-gradient, and AdamW arithmetic in a
/// step is dispatched as Vulkan compute. CPU code only uploads the minibatch,
/// submits work, and reads scalar diagnostics/checkpoints.
pub struct HierarchosHeadTrainer {
    device: VulkanDevice,
    context_dim: usize,
    vocab_size: usize,
    max_rows: usize,
    lm_head: SharedLmHeadParameter,
    hidden: GpuBuffer,
    targets: GpuBuffer,
    logits: GpuBuffer,
    ce_row_stats: GpuBuffer,
    row_loss: GpuBuffer,
    row_loss_readback: GpuBuffer,
    grad_weight: GpuBuffer,
    linear_forward: vulkan::ComputeKernel,
    cross_entropy: vulkan::ComputeKernel,
    cross_entropy_linear_weight_grad: vulkan::ComputeKernel,
}

impl HierarchosHeadTrainer {
    pub fn new(
        device: VulkanDevice,
        context_dim: usize,
        vocab_size: usize,
        max_rows: usize,
        weight: &[f32],
    ) -> Result<Self> {
        if context_dim == 0 || vocab_size == 0 || max_rows == 0 {
            bail!("context_dim, vocab_size, and max_rows must be positive");
        }
        let weight_len = context_dim
            .checked_mul(vocab_size)
            .context("weight element count overflow")?;
        if weight.len() != weight_len {
            bail!(
                "lm_head weight has {} values; expected {} for shape [{}, {}]",
                weight.len(),
                weight_len,
                vocab_size,
                context_dim
            );
        }
        if weight.iter().any(|value| !value.is_finite()) {
            bail!("lm_head weight contains non-finite values");
        }

        let lm_head = SharedLmHeadParameter::new(device, context_dim, vocab_size, weight)?;
        Self::from_shared_lm_head(lm_head, max_rows)
    }

    /// Bind the LM-head loss to an existing shared tied parameter. This is the
    /// composition point used when recurrent DeepEmbed branches and the loss
    /// must contribute to one `lm_head.weight` optimizer state.
    pub fn from_shared_lm_head(lm_head: SharedLmHeadParameter, max_rows: usize) -> Result<Self> {
        if max_rows == 0 {
            bail!("max_rows must be positive");
        }
        let device = lm_head.device();
        let context_dim = lm_head.context_dim();
        let vocab_size = lm_head.vocab_size();
        let weight_len = context_dim
            .checked_mul(vocab_size)
            .context("weight element count overflow")?;

        let logits_len = max_rows
            .checked_mul(vocab_size)
            .context("logit element count overflow")?;
        let hidden_len = max_rows
            .checked_mul(context_dim)
            .context("hidden element count overflow")?;

        Ok(Self {
            linear_forward: vulkan::ComputeKernel::new(
                &device,
                LINEAR_FORWARD_SPV,
                3,
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            cross_entropy: vulkan::ComputeKernel::new(
                &device,
                CROSS_ENTROPY_ROW_STATS_SPV,
                4,
                std::mem::size_of::<CrossEntropyPush>() as u32,
            )?,
            cross_entropy_linear_weight_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<WeightGradPush>() as u32,
            )?,
            lm_head,
            hidden: GpuBuffer::zeros_f32(&device, hidden_len)?,
            targets: GpuBuffer::zeros_u32(&device, max_rows)?,
            logits: GpuBuffer::zeros_f32(&device, logits_len)?,
            ce_row_stats: GpuBuffer::zeros_f32(&device, max_rows * 3)?,
            row_loss: GpuBuffer::zeros_f32(&device, max_rows)?,
            row_loss_readback: GpuBuffer::zeros_host_f32(&device, max_rows)?,
            grad_weight: GpuBuffer::zeros_f32(&device, weight_len)?,
            device,
            context_dim,
            vocab_size,
            max_rows,
        })
    }

    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        max_rows: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = ModelConfig::from_model_dir(model_dir)
            .context("validating Hierarchos native model contract")?;
        let (shape, weight) =
            read_f32_tensor(&model_dir.join("model.safetensors"), "lm_head.weight")?;
        let expected = vec![config.vocab_size, config.context_dim];
        if shape != expected {
            bail!("lm_head.weight has shape {shape:?}; expected {expected:?}");
        }
        Self::new(
            device,
            config.context_dim,
            config.vocab_size,
            max_rows,
            &weight,
        )
    }

    pub fn train_step(
        &mut self,
        hidden: &[f32],
        targets: &[u32],
        hyper: AdamWHyperParams,
    ) -> Result<TrainStepResult> {
        self.train_step_internal(hidden, targets, hyper, true)
    }

    /// Add the dense LM-loss gradient to an already-populated shared
    /// `lm_head.weight` accumulator, then perform the single final AdamW step.
    pub fn train_step_finalize_shared_lm(
        &mut self,
        hidden: &[f32],
        targets: &[u32],
        hyper: AdamWHyperParams,
    ) -> Result<TrainStepResult> {
        self.train_step_internal(hidden, targets, hyper, false)
    }

    fn train_step_internal(
        &mut self,
        hidden: &[f32],
        targets: &[u32],
        hyper: AdamWHyperParams,
        reset_shared_lm_grad: bool,
    ) -> Result<TrainStepResult> {
        hyper.validate()?;
        if !hidden.len().is_multiple_of(self.context_dim) {
            bail!(
                "hidden length {} is not divisible by context_dim {}",
                hidden.len(),
                self.context_dim
            );
        }
        let rows = hidden.len() / self.context_dim;
        if rows == 0 || rows > self.max_rows {
            bail!(
                "batch has {rows} rows; trainer capacity is 1..={}",
                self.max_rows
            );
        }
        if targets.len() != rows {
            bail!(
                "target count {} does not match batch rows {rows}",
                targets.len()
            );
        }
        if let Some(&bad) = targets
            .iter()
            .find(|&&target| target as usize >= self.vocab_size)
        {
            bail!(
                "target token {bad} is outside vocabulary size {}",
                self.vocab_size
            );
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            bail!("hidden batch contains non-finite values");
        }

        let mut batch = vulkan::ComputeBatch::new(&self.device)?;
        batch.upload_f32(&self.hidden, hidden)?;
        batch.upload_u32(&self.targets, targets)?;
        if reset_shared_lm_grad {
            self.lm_head.record_zero_grad(&mut batch)?;
        }
        let linear_push = LinearPush {
            rows: rows as u32,
            input_dim: self.context_dim as u32,
            output_dim: self.vocab_size as u32,
            z_loss_weight: 0.0,
            activation_clamp: f32::MAX,
        };
        self.linear_forward.record_dispatch(
            &mut batch,
            &[&self.hidden, self.lm_head.weight_buffer(), &self.logits],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(self.vocab_size, 16), div_ceil_u32(rows, 16), 1],
        )?;

        let xent_push = CrossEntropyPush {
            rows: rows as u32,
            vocab_size: self.vocab_size as u32,
            z_loss_weight: 0.0,
        };
        self.cross_entropy.record_dispatch(
            &mut batch,
            &[
                &self.logits,
                &self.targets,
                &self.ce_row_stats,
                &self.row_loss,
            ],
            bytemuck::bytes_of(&xent_push),
            [rows as u32, 1, 1],
        )?;

        let grad_push = WeightGradPush {
            rows: rows as u32,
            input_dim: self.context_dim as u32,
            output_dim: self.vocab_size as u32,
        };
        self.cross_entropy_linear_weight_grad.record_dispatch(
            &mut batch,
            &[
                &self.hidden,
                &self.logits,
                &self.ce_row_stats,
                &self.grad_weight,
            ],
            bytemuck::bytes_of(&grad_push),
            [
                div_ceil_u32(self.context_dim, 16),
                div_ceil_u32(self.vocab_size, 16),
                1,
            ],
        )?;
        self.lm_head
            .record_accumulate_gradient(&mut batch, &self.grad_weight)?;
        let next_step = self.lm_head.record_step(&mut batch, hyper)?;
        batch.readback_f32(&self.row_loss, &self.row_loss_readback, rows)?;
        batch.submit()?;

        let losses = self.row_loss_readback.read_f32(rows)?;
        let loss = losses.iter().sum::<f32>() / rows as f32;
        if !loss.is_finite() {
            bail!("Vulkan training step produced non-finite loss");
        }
        Ok(TrainStepResult {
            loss,
            step: next_step,
        })
    }

    pub fn weights(&self) -> Result<Vec<f32>> {
        self.lm_head.weights()
    }

    pub fn shared_lm_head(&self) -> SharedLmHeadParameter {
        self.lm_head.clone()
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn export_model_package(
        &self,
        source_model_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
    ) -> Result<()> {
        let source_model_dir = source_model_dir.as_ref();
        let output_dir = output_dir.as_ref();
        if source_model_dir == output_dir {
            bail!("export_model_package requires a distinct output directory");
        }
        let config = ModelConfig::from_model_dir(source_model_dir)
            .context("validating source model package")?;
        if config.context_dim != self.context_dim || config.vocab_size != self.vocab_size {
            bail!(
                "trainer shape [{}, {}] does not match model package [{}, {}]",
                self.vocab_size,
                self.context_dim,
                config.vocab_size,
                config.context_dim
            );
        }
        std::fs::create_dir_all(output_dir)?;
        for entry in std::fs::read_dir(source_model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path.file_name().and_then(|name| name.to_str()) != Some("model.safetensors")
            {
                std::fs::copy(&path, output_dir.join(entry.file_name()))?;
            }
        }
        let weights = self.weights()?;
        replace_f32_tensor(
            &source_model_dir.join("model.safetensors"),
            &output_dir.join("model.safetensors"),
            "lm_head.weight",
            &[self.vocab_size, self.context_dim],
            &weights,
        )?;
        Ok(())
    }
}

/// Vulkan training slice that extends the LM-head parity beachhead backward
/// through Hierarchos' real `out_norm` LayerNorm. The caller supplies the
/// pre-normalization activations produced by the recurrent body; Vulkan owns
/// LayerNorm forward/backward, LM-head forward/backward, loss, and AdamW.
///
/// This deliberately preserves the Python optimizer contract: the 2-D tied
/// token/LM-head matrix receives decoupled weight decay, while LayerNorm's 1-D
/// weight and bias use the same AdamW moments with zero weight decay.
pub struct HierarchosOutNormHeadTrainer {
    device: VulkanDevice,
    context_dim: usize,
    vocab_size: usize,
    max_rows: usize,
    activation_clamp: f32,
    step: u32,
    lm_execution_arm: HierarchosLmExecutionArm,
    lm_weight_grad_topology: HierarchosLmWeightGradTopology,
    lm_fused_adjoint_topology: lm_execution::HierarchosLmFusedAdjointTopology,
    native_fp16_lm_backward_compute: bool,
    native_fp16_lm_input_grad_compute: bool,
    native_fp16_out_norm_backward_compute: bool,
    lm_head: SharedLmHeadParameter,
    norm_weight: GpuBuffer,
    norm_bias: GpuBuffer,
    input_hidden: GpuBuffer,
    norm_mean: GpuBuffer,
    norm_rstd: GpuBuffer,
    targets: GpuBuffer,
    ce_row_stats: GpuBuffer,
    ce_grad_tape: GpuBuffer,
    ce_tile_stats: GpuBuffer,
    ce_input_grad_partials: Option<GpuBuffer>,
    row_loss: GpuBuffer,
    row_loss_readback: GpuBuffer,
    grad_lm_weight: GpuBuffer,
    grad_norm_hidden: GpuBuffer,
    grad_input_hidden: GpuBuffer,
    grad_norm_weight: GpuBuffer,
    grad_norm_bias: GpuBuffer,
    tied_token_ids: GpuBuffer,
    tied_sorted_token_positions: GpuBuffer,
    tied_radix_scratch_positions: GpuBuffer,
    tied_radix_block_histograms: GpuBuffer,
    tied_radix_block_offsets: GpuBuffer,
    tied_embedding_grad: GpuBuffer,
    norm_weight_exp_avg: GpuBuffer,
    norm_weight_exp_avg_sq: GpuBuffer,
    norm_bias_exp_avg: GpuBuffer,
    norm_bias_exp_avg_sq: GpuBuffer,
    layer_norm_stats: vulkan::ComputeKernel,
    layer_norm_affine_clamp_backward_inplace: vulkan::ComputeKernel,
    cross_entropy_linear_row_stats_streaming: vulkan::ComputeKernel,
    cross_entropy_linear_row_stats_streaming_fp16_packed: vulkan::ComputeKernel,
    cross_entropy_linear_row_stats_streaming_fp16_packed_tape: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_logit_tape_fp16_packed_rows8: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_logit_tape_fp16_packed_rows16: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_logit_tape_fp16_packed_rows16_dot4: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_logit_tape_fp16_packed_rows16_cluster4: Option<vulkan::ComputeKernel>,
    cross_entropy_row_stats_tile_partials: vulkan::ComputeKernel,
    cross_entropy_row_stats_tile_partials_rows16: vulkan::ComputeKernel,
    cross_entropy_linear_weight_grad_streaming: vulkan::ComputeKernel,
    cross_entropy_linear_weight_grad_streaming_fp16_packed_rows4: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_streaming_fp16_packed: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_streaming_fp16_packed_rows16: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows4:
        Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_streaming_fp16_native_compute: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows16:
        Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_tape_fp16_packed_rows4: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_tape_fp16_packed: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_weight_grad_tape_fp16_packed_rows16: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_streaming: vulkan::ComputeKernel,
    cross_entropy_linear_input_grad_streaming_fp16_packed: vulkan::ComputeKernel,
    cross_entropy_linear_input_grad_tape_fp16_packed: vulkan::ComputeKernel,
    cross_entropy_linear_adjoints_tape_fp16_packed_fused: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden:
        Option<vulkan::ComputeKernel>,
    cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256:
        Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_tile_reduce: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_tile_reduce_tile256: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_streaming_fp16_native: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_streaming_fp16_native_reuse64: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_streaming_fp16_native_reuse128: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_streaming_fp16_native_reuse224: Option<vulkan::ComputeKernel>,
    cross_entropy_linear_input_grad_streaming_fp16_native_compute: Option<vulkan::ComputeKernel>,
    cross_entropy_row_loss_extract: vulkan::ComputeKernel,
    layer_norm_input_grad: vulkan::ComputeKernel,
    layer_norm_input_grad_fp16_native_compute: Option<vulkan::ComputeKernel>,
    layer_norm_param_grad: vulkan::ComputeKernel,
    layer_norm_param_grad_fp16_native_compute: Option<vulkan::ComputeKernel>,
    embedding_token_sort: vulkan::ComputeKernel,
    embedding_radix_histogram: vulkan::ComputeKernel,
    embedding_radix_prefix: vulkan::ComputeKernel,
    embedding_radix_scatter: vulkan::ComputeKernel,
    embedding_grad_segmented: vulkan::ComputeKernel,
    adamw: vulkan::ComputeKernel,
}

#[derive(Debug)]
pub(crate) struct HierarchosOutNormRecordedStep {
    rows: usize,
    lm_step: u32,
    next_norm_step: u32,
}

#[derive(Debug)]
pub(crate) struct HierarchosOutNormBackwardTicket {
    rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LmWeightGradWriteMode {
    ScratchOverwrite,
    SharedOverwrite,
    SharedAccumulate,
}

impl LmWeightGradWriteMode {
    fn uses_shared_buffer(self) -> bool {
        !matches!(self, Self::ScratchOverwrite)
    }

    fn accumulates_existing(self) -> bool {
        matches!(self, Self::SharedAccumulate)
    }
}

impl HierarchosOutNormHeadTrainer {
    const LAYER_NORM_EPS: f32 = 1.0e-5;
    const DEFAULT_ACTIVATION_CLAMP: f32 = 100.0;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: VulkanDevice,
        context_dim: usize,
        vocab_size: usize,
        max_rows: usize,
        lm_weight: &[f32],
        norm_weight: &[f32],
        norm_bias: &[f32],
    ) -> Result<Self> {
        if context_dim == 0 || vocab_size == 0 || max_rows == 0 {
            bail!("context_dim, vocab_size, and max_rows must be positive");
        }
        let lm_len = context_dim
            .checked_mul(vocab_size)
            .context("LM-head weight element count overflow")?;
        if lm_weight.len() != lm_len {
            bail!(
                "lm_head weight has {} values; expected {} for shape [{}, {}]",
                lm_weight.len(),
                lm_len,
                vocab_size,
                context_dim
            );
        }
        if norm_weight.len() != context_dim || norm_bias.len() != context_dim {
            bail!(
                "out_norm vectors must both have context_dim {} values; got weight={} bias={}",
                context_dim,
                norm_weight.len(),
                norm_bias.len()
            );
        }
        if lm_weight
            .iter()
            .chain(norm_weight)
            .chain(norm_bias)
            .any(|value| !value.is_finite())
        {
            bail!("out_norm/LM-head parameters contain non-finite values");
        }

        let lm_head = SharedLmHeadParameter::new(device, context_dim, vocab_size, lm_weight)?;
        Self::from_shared_lm_head(lm_head, max_rows, norm_weight, norm_bias)
    }

    pub fn from_shared_lm_head(
        lm_head: SharedLmHeadParameter,
        max_rows: usize,
        norm_weight: &[f32],
        norm_bias: &[f32],
    ) -> Result<Self> {
        if max_rows == 0 {
            bail!("max_rows must be positive");
        }
        let device = lm_head.device();
        let context_dim = lm_head.context_dim();
        let vocab_size = lm_head.vocab_size();
        let lm_len = context_dim
            .checked_mul(vocab_size)
            .context("LM-head weight element count overflow")?;
        if norm_weight.len() != context_dim || norm_bias.len() != context_dim {
            bail!(
                "out_norm vectors must both have context_dim {} values; got weight={} bias={}",
                context_dim,
                norm_weight.len(),
                norm_bias.len()
            );
        }
        if norm_weight
            .iter()
            .chain(norm_bias)
            .any(|value| !value.is_finite())
        {
            bail!("out_norm parameters contain non-finite values");
        }

        let hidden_len = max_rows
            .checked_mul(context_dim)
            .context("hidden element count overflow")?;
        let tied_radix_block_capacity = max_rows.div_ceil(EMBEDDING_RADIX_BLOCK_SIZE);
        let tied_radix_scratch_len = tied_radix_block_capacity
            .checked_mul(EMBEDDING_RADIX_BUCKETS)
            .context("tied embedding radix scratch element count overflow")?;
        let ce_grad_tape_len = max_rows
            .checked_mul(vocab_size)
            .context("CE gradient tape element count overflow")?;
        let ce_tile_stats_len = max_rows
            .checked_mul(vocab_size.div_ceil(8))
            .and_then(|count| count.checked_mul(3))
            .context("CE tile-stat element count overflow")?;
        let ce_input_grad_partial_len = vocab_size
            .div_ceil(CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_VOCAB_TILE)
            .checked_mul(max_rows)
            .and_then(|count| count.checked_mul(context_dim))
            .context("CE fused-adjoint input-gradient partial count overflow")?;
        let fused_adjoints_supported = context_dim <= 448
            && max_rows <= CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_MAX_ROWS
            && device.max_compute_shared_memory_bytes()
                >= CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_SHARED_BYTES
            && device.supports_compute_work_group_size_x(64)
            && device.supports_storage_buffer_bindings(10);

        Ok(Self {
            layer_norm_stats: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_STATS_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LayerNormForwardPush>() as u32,
            )?,
            layer_norm_affine_clamp_backward_inplace: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_AFFINE_CLAMP_BACKWARD_INPLACE_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadWrite,
                ],
                std::mem::size_of::<LayerNormAffineClampBackwardPush>() as u32,
            )?,
            cross_entropy_linear_row_stats_streaming: vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            cross_entropy_linear_row_stats_streaming_fp16_packed:
                vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?,
            cross_entropy_linear_row_stats_streaming_fp16_packed_tape: if device
                .supports_storage_buffer_bindings(9)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_TAPE_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_linear_logit_tape_fp16_packed_rows8: if context_dim <= 448
                && device.max_compute_shared_memory_bytes() >= 16_160
                && device.supports_compute_work_group_size_x(64)
                && device.supports_storage_buffer_bindings(9)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS8_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_linear_logit_tape_fp16_packed_rows16: if context_dim <= 448
                && device.max_compute_shared_memory_bytes() >= 16_192
                && device.supports_compute_work_group_size_x(64)
                && device.supports_storage_buffer_bindings(9)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_linear_logit_tape_fp16_packed_rows16_dot4: if context_dim <= 448
                && device.max_compute_shared_memory_bytes() >= 16_384
                && device.supports_compute_work_group_size_x(64)
                && device.supports_storage_buffer_bindings(9)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_DOT4_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_linear_logit_tape_fp16_packed_rows16_cluster4: if context_dim <= 448
                && device.max_compute_shared_memory_bytes() >= 16_192
                && device.supports_compute_work_group_size_x(64)
                && device.supports_storage_buffer_bindings(9)
                && device.supports_compute_subgroup_clustered_arithmetic()
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_CLUSTER4_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_row_stats_tile_partials: vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            cross_entropy_row_stats_tile_partials_rows16: vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_ROWS16_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            cross_entropy_linear_weight_grad_streaming: vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadWrite,
                ],
                std::mem::size_of::<LmWeightGradPush>() as u32,
            )?,
            cross_entropy_linear_weight_grad_streaming_fp16_packed_rows4:
                create_fp16_lm_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows4,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS4_SPV,
                )?,
            cross_entropy_linear_weight_grad_streaming_fp16_packed:
                create_fp16_lm_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows8,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_SPV,
                )?,
            cross_entropy_linear_weight_grad_streaming_fp16_packed_rows16:
                create_fp16_lm_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows16,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS16_SPV,
                )?,
            cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows4:
                create_native_fp16_lm_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows4,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_ROWS4_SPV,
                )?,
            cross_entropy_linear_weight_grad_streaming_fp16_native_compute:
                create_native_fp16_lm_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows8,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_SPV,
                )?,
            cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows16:
                create_native_fp16_lm_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows16,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_ROWS16_SPV,
                )?,
            cross_entropy_linear_weight_grad_tape_fp16_packed_rows4:
                create_fp16_lm_tape_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows4,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS4_SPV,
                )?,
            cross_entropy_linear_weight_grad_tape_fp16_packed:
                create_fp16_lm_tape_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows8,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_SPV,
                )?,
            cross_entropy_linear_weight_grad_tape_fp16_packed_rows16:
                create_fp16_lm_tape_weight_grad_kernel(
                    &device,
                    HierarchosLmWeightGradTopology::VocabRows16,
                    CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS16_SPV,
                )?,
            cross_entropy_linear_input_grad_streaming: vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            cross_entropy_linear_input_grad_streaming_fp16_packed:
                vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_PACKED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?,
            cross_entropy_linear_input_grad_tape_fp16_packed:
                vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_TAPE_FP16_PACKED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?,
            cross_entropy_linear_adjoints_tape_fp16_packed_fused: if fused_adjoints_supported {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LmWeightGradPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden:
                if fused_adjoints_supported {
                    Some(vulkan::ComputeKernel::new_with_access(
                        &device,
                        CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadWrite,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<LmWeightGradPush>() as u32,
                    )?)
                } else {
                    None
                },
            cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256:
                if fused_adjoints_supported && device.supports_compute_work_group_size_x(256) {
                    Some(vulkan::ComputeKernel::new_with_access(
                        &device,
                        CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_WG256_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadWrite,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<LmWeightGradPush>() as u32,
                    )?)
                } else {
                    None
                },
            cross_entropy_linear_input_grad_tile_reduce: if fused_adjoints_supported {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_linear_input_grad_tile_reduce_tile256: if fused_adjoints_supported
                && device.supports_compute_work_group_size_x(256)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE256_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            cross_entropy_linear_input_grad_streaming_fp16_native:
                create_native_fp16_lm_input_grad_kernel(
                    &device,
                    HierarchosLmExecutionArm::Fp16Native,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_SPV,
                )?,
            cross_entropy_linear_input_grad_streaming_fp16_native_reuse64:
                create_native_fp16_lm_input_grad_kernel(
                    &device,
                    HierarchosLmExecutionArm::Fp16NativeReuse64,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE64_SPV,
                )?,
            cross_entropy_linear_input_grad_streaming_fp16_native_reuse128:
                create_native_fp16_lm_input_grad_kernel(
                    &device,
                    HierarchosLmExecutionArm::Fp16NativeReuse128,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE128_SPV,
                )?,
            cross_entropy_linear_input_grad_streaming_fp16_native_reuse224:
                create_native_fp16_lm_input_grad_kernel(
                    &device,
                    HierarchosLmExecutionArm::Fp16NativeReuse224,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE224_SPV,
                )?,
            cross_entropy_linear_input_grad_streaming_fp16_native_compute:
                create_native_fp16_lm_input_grad_kernel(
                    &device,
                    HierarchosLmExecutionArm::Fp16Native,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_COMPUTE_SPV,
                )?,
            cross_entropy_row_loss_extract: vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_ROW_LOSS_EXTRACT_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<CrossEntropyPush>() as u32,
            )?,
            layer_norm_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_INPUT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LayerNormBackwardPush>() as u32,
            )?,
            layer_norm_input_grad_fp16_native_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        LAYER_NORM_INPUT_GRAD_FP16_NATIVE_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<LayerNormBackwardPush>() as u32,
                    )
                })
                .transpose()?,
            layer_norm_param_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_PARAM_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LayerNormBackwardPush>() as u32,
            )?,
            layer_norm_param_grad_fp16_native_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        LAYER_NORM_PARAM_GRAD_FP16_NATIVE_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<LayerNormBackwardPush>() as u32,
                    )
                })
                .transpose()?,
            embedding_token_sort: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING_TOKEN_SORT_SPV,
                2,
                std::mem::size_of::<EmbeddingGradPush>() as u32,
            )?,
            embedding_radix_histogram: vulkan::ComputeKernel::new_with_access(
                &device,
                EMBEDDING_RADIX_HISTOGRAM_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<EmbeddingRadixPush>() as u32,
            )?,
            embedding_radix_prefix: vulkan::ComputeKernel::new_with_access(
                &device,
                EMBEDDING_RADIX_PREFIX_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<EmbeddingRadixPush>() as u32,
            )?,
            embedding_radix_scatter: vulkan::ComputeKernel::new_with_access(
                &device,
                EMBEDDING_RADIX_SCATTER_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<EmbeddingRadixPush>() as u32,
            )?,
            embedding_grad_segmented: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING_GRAD_SEGMENTED_SPV,
                4,
                std::mem::size_of::<EmbeddingGradPush>() as u32,
            )?,
            adamw: vulkan::ComputeKernel::new(
                &device,
                ADAMW_SPV,
                4,
                std::mem::size_of::<AdamWPush>() as u32,
            )?,
            lm_head,
            norm_weight: GpuBuffer::from_f32(&device, norm_weight)?,
            norm_bias: GpuBuffer::from_f32(&device, norm_bias)?,
            input_hidden: GpuBuffer::zeros_f32(&device, hidden_len)?,
            norm_mean: GpuBuffer::zeros_f32(&device, max_rows)?,
            norm_rstd: GpuBuffer::zeros_f32(&device, max_rows)?,
            targets: GpuBuffer::zeros_u32(&device, max_rows)?,
            ce_row_stats: GpuBuffer::zeros_f32(&device, max_rows * 5)?,
            // Reusable scratch for packed-FP16 loss arms. Streaming backward
            // can write final FP32 CE adjoints here, while CE-tape arms keep
            // forward logits as a parity anchor and derive adjoints only at
            // their W^T/dW consumers. Cross-row tape arms let each resident LM
            // weight tile serve multiple Hierarchos rows before eviction. Their
            // compact tile-stat scratch carries max/scaled-sum/target-logit
            // partials so CE stats do not reread the full vocabulary tape.
            ce_grad_tape: GpuBuffer::zeros_f32(&device, ce_grad_tape_len)?,
            ce_tile_stats: GpuBuffer::zeros_f32(&device, ce_tile_stats_len)?,
            ce_input_grad_partials: if fused_adjoints_supported {
                Some(GpuBuffer::zeros_f32(&device, ce_input_grad_partial_len)?)
            } else {
                None
            },
            row_loss: GpuBuffer::zeros_f32(&device, max_rows)?,
            row_loss_readback: GpuBuffer::zeros_host_f32(&device, max_rows)?,
            grad_lm_weight: GpuBuffer::zeros_f32(&device, lm_len)?,
            grad_norm_hidden: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_input_hidden: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_norm_weight: GpuBuffer::zeros_f32(&device, context_dim)?,
            grad_norm_bias: GpuBuffer::zeros_f32(&device, context_dim)?,
            tied_token_ids: GpuBuffer::zeros_u32(&device, max_rows)?,
            tied_sorted_token_positions: GpuBuffer::zeros_u32(&device, max_rows)?,
            tied_radix_scratch_positions: GpuBuffer::zeros_u32(&device, max_rows)?,
            tied_radix_block_histograms: GpuBuffer::zeros_u32(&device, tied_radix_scratch_len)?,
            tied_radix_block_offsets: GpuBuffer::zeros_u32(&device, tied_radix_scratch_len)?,
            tied_embedding_grad: GpuBuffer::zeros_f32(&device, hidden_len)?,
            norm_weight_exp_avg: GpuBuffer::zeros_f32(&device, context_dim)?,
            norm_weight_exp_avg_sq: GpuBuffer::zeros_f32(&device, context_dim)?,
            norm_bias_exp_avg: GpuBuffer::zeros_f32(&device, context_dim)?,
            norm_bias_exp_avg_sq: GpuBuffer::zeros_f32(&device, context_dim)?,
            device,
            context_dim,
            vocab_size,
            max_rows,
            activation_clamp: Self::DEFAULT_ACTIVATION_CLAMP,
            step: 0,
            lm_execution_arm: HierarchosLmExecutionArm::Fp32,
            lm_weight_grad_topology: HierarchosLmWeightGradTopology::VocabRows8,
            lm_fused_adjoint_topology: lm_execution::HierarchosLmFusedAdjointTopology::SharedHidden,
            native_fp16_lm_backward_compute: false,
            native_fp16_lm_input_grad_compute: false,
            native_fp16_out_norm_backward_compute: false,
        })
    }

    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        max_rows: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = ModelConfig::from_model_dir(model_dir)
            .context("validating Hierarchos native model contract")?;
        let tensor_path = model_dir.join("model.safetensors");
        let (lm_shape, lm_weight) = read_f32_tensor(&tensor_path, "lm_head.weight")?;
        let (norm_weight_shape, norm_weight) = read_f32_tensor(&tensor_path, "out_norm.weight")?;
        let (norm_bias_shape, norm_bias) = read_f32_tensor(&tensor_path, "out_norm.bias")?;
        let expected_lm = vec![config.vocab_size, config.context_dim];
        let expected_norm = vec![config.context_dim];
        if lm_shape != expected_lm {
            bail!("lm_head.weight has shape {lm_shape:?}; expected {expected_lm:?}");
        }
        if norm_weight_shape != expected_norm || norm_bias_shape != expected_norm {
            bail!(
                "out_norm tensors have shapes weight={norm_weight_shape:?} bias={norm_bias_shape:?}; expected {expected_norm:?}"
            );
        }
        let mut trainer = Self::new(
            device,
            config.context_dim,
            config.vocab_size,
            max_rows,
            &lm_weight,
            &norm_weight,
            &norm_bias,
        )?;
        trainer.set_activation_clamp(config.activation_clamp)?;
        Ok(trainer)
    }

    /// Select the same finite-preserving output activation ceiling used by
    /// `HierarchosCore._finite_clamp(out_norm(...))`. The ceiling is carried
    /// through every streamed LM reconstruction; no global out_norm tensor is
    /// materialized.
    pub fn set_activation_clamp(&mut self, max_abs: f32) -> Result<()> {
        if !max_abs.is_finite() || max_abs <= 0.0 {
            bail!("out_norm activation clamp must be finite and positive; got {max_abs}");
        }
        self.activation_clamp = max_abs;
        Ok(())
    }

    pub fn activation_clamp(&self) -> f32 {
        self.activation_clamp
    }

    pub fn train_step(
        &mut self,
        pre_norm_hidden: &[f32],
        targets: &[u32],
        hyper: AdamWHyperParams,
    ) -> Result<TrainStepResult> {
        self.train_step_internal(pre_norm_hidden, targets, None, hyper, true)
    }

    /// Finalize an LM-head update after recurrent branches have already
    /// accumulated tied-embedding gradients into the shared parameter. This
    /// preserves those gradients, adds the dense cross-entropy contribution,
    /// and performs the single shared AdamW step.
    pub fn train_step_finalize_shared_lm(
        &mut self,
        pre_norm_hidden: &[f32],
        targets: &[u32],
        hyper: AdamWHyperParams,
    ) -> Result<TrainStepResult> {
        self.train_step_internal(pre_norm_hidden, targets, None, hyper, false)
    }

    /// Record the dense output loss after other users of the tied
    /// `lm_head.weight` have already accumulated their gradients. Queue
    /// submission is deliberately left to the caller so H-RWKV, L-RWKV, and
    /// the loss can execute in one command buffer.
    pub(crate) fn record_finalize_shared_lm(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &[f32],
        targets: &[u32],
        hyper: AdamWHyperParams,
    ) -> Result<HierarchosOutNormRecordedStep> {
        self.record_train_step_internal(batch, pre_norm_hidden, targets, None, hyper, false)
    }

    pub(crate) fn finalize_recorded_train_step(
        &mut self,
        recorded: HierarchosOutNormRecordedStep,
    ) -> Result<TrainStepResult> {
        let losses = self.row_loss_readback.read_f32(recorded.rows)?;
        let loss = losses.iter().sum::<f32>() / recorded.rows as f32;
        if !loss.is_finite() {
            bail!("Vulkan out_norm training step produced non-finite loss");
        }
        self.step = recorded.next_norm_step;
        Ok(TrainStepResult {
            loss,
            step: recorded.lm_step,
        })
    }

    /// Record only the row statistics required to reconstruct `out_norm`
    /// inside the streaming LM kernels. The affine `[rows, context_dim]`
    /// activation is intentionally never materialized globally.
    fn record_norm_head_stats(
        &self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
    ) -> Result<LinearPush> {
        let linear_push = LinearPush {
            rows: rows as u32,
            input_dim: self.context_dim as u32,
            output_dim: self.vocab_size as u32,
            z_loss_weight: 0.0,
            activation_clamp: self.activation_clamp,
        };
        let norm_forward_push = LayerNormForwardPush {
            rows: rows as u32,
            dim: self.context_dim as u32,
            eps: Self::LAYER_NORM_EPS,
        };
        self.layer_norm_stats.record_dispatch(
            batch,
            &[pre_norm_hidden, &self.norm_mean, &self.norm_rstd],
            bytemuck::bytes_of(&norm_forward_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        Ok(linear_push)
    }

    pub fn lm_execution_arm(&self) -> HierarchosLmExecutionArm {
        self.lm_execution_arm
    }

    pub fn lm_weight_grad_topology(&self) -> Option<HierarchosLmWeightGradTopology> {
        self.lm_execution_arm
            .uses_fp16_weights()
            .then_some(self.lm_weight_grad_topology)
    }

    pub fn lm_fused_adjoint_topology_label(&self) -> Option<&'static str> {
        self.lm_execution_arm
            .fuses_ce_adjoints()
            .then_some(self.lm_fused_adjoint_topology.label())
    }

    pub fn native_fp16_lm_backward_compute_active(&self) -> bool {
        self.native_fp16_lm_backward_compute
    }

    /// Whether the LM W^T/input-adjoint leg itself is executing native FP16
    /// products. The native dW leg may remain enabled while this returns false;
    /// that split is useful for parity-sensitive training because tiny CE
    /// adjoints can change sign after an FP16 round-trip even when the much
    /// larger parameter-gradient products remain well behaved.
    pub fn native_fp16_lm_input_grad_compute_active(&self) -> bool {
        self.native_fp16_lm_input_grad_compute
    }

    pub fn native_fp16_out_norm_backward_compute_active(&self) -> bool {
        self.native_fp16_out_norm_backward_compute
    }

    fn fp16_lm_input_grad_kernel(
        &self,
        arm: HierarchosLmExecutionArm,
    ) -> Result<&vulkan::ComputeKernel> {
        match arm {
            HierarchosLmExecutionArm::Fp16Packed => {
                Ok(&self.cross_entropy_linear_input_grad_streaming_fp16_packed)
            }
            HierarchosLmExecutionArm::Fp16CeTape
            | HierarchosLmExecutionArm::Fp16CeTapeRows8
            | HierarchosLmExecutionArm::Fp16CeTapeRows16
            | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
            | HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints
            | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints
            | HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints => {
                bail!("FP16 CE-tape input adjoint uses the dedicated tape W^T kernel")
            }
            HierarchosLmExecutionArm::Fp16Native
                if self.native_fp16_lm_input_grad_compute_active() =>
            {
                self.cross_entropy_linear_input_grad_streaming_fp16_native_compute
                    .as_ref()
                    .context("native-FP16-compute LM input-adjoint arm is unavailable")
            }
            HierarchosLmExecutionArm::Fp16Native => self
                .cross_entropy_linear_input_grad_streaming_fp16_native
                .as_ref()
                .context("native-FP16 LM input-adjoint reuse32 arm is unavailable"),
            HierarchosLmExecutionArm::Fp16NativeReuse64 => self
                .cross_entropy_linear_input_grad_streaming_fp16_native_reuse64
                .as_ref()
                .context("native-FP16 LM input-adjoint reuse64 arm is unavailable"),
            HierarchosLmExecutionArm::Fp16NativeReuse128 => self
                .cross_entropy_linear_input_grad_streaming_fp16_native_reuse128
                .as_ref()
                .context("native-FP16 LM input-adjoint reuse128 arm is unavailable"),
            HierarchosLmExecutionArm::Fp16NativeReuse224 => self
                .cross_entropy_linear_input_grad_streaming_fp16_native_reuse224
                .as_ref()
                .context("native-FP16 LM input-adjoint reuse224 arm is unavailable"),
            HierarchosLmExecutionArm::Fp32 => {
                bail!("FP32 is not an FP16 LM input-adjoint arm")
            }
        }
    }

    fn fp16_ce_cross_row_kernels(
        &self,
        arm: HierarchosLmExecutionArm,
    ) -> Result<(&vulkan::ComputeKernel, &vulkan::ComputeKernel, usize)> {
        match arm {
            HierarchosLmExecutionArm::Fp16CeTapeRows8 => Ok((
                self.cross_entropy_linear_logit_tape_fp16_packed_rows8
                    .as_ref()
                    .context("FP16 CE-tape rows8 cross-row projection kernel is unavailable")?,
                &self.cross_entropy_row_stats_tile_partials,
                8,
            )),
            HierarchosLmExecutionArm::Fp16CeTapeRows16
            | HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints => Ok((
                self.cross_entropy_linear_logit_tape_fp16_packed_rows16
                    .as_ref()
                    .context("FP16 CE-tape rows16 cross-row projection kernel is unavailable")?,
                &self.cross_entropy_row_stats_tile_partials_rows16,
                16,
            )),
            HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4 => Ok((
                self.cross_entropy_linear_logit_tape_fp16_packed_rows16_dot4
                    .as_ref()
                    .context(
                        "FP16 CE-tape rows16-dot4 cross-row projection kernel is unavailable",
                    )?,
                &self.cross_entropy_row_stats_tile_partials_rows16,
                16,
            )),
            HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints => Ok((
                self.cross_entropy_linear_logit_tape_fp16_packed_rows16_dot4
                    .as_ref()
                    .context(
                        "FP16 CE-tape rows16-dot4 cross-row projection kernel is unavailable",
                    )?,
                &self.cross_entropy_row_stats_tile_partials_rows16,
                16,
            )),
            HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints => Ok((
                self.cross_entropy_linear_logit_tape_fp16_packed_rows16_cluster4
                    .as_ref()
                    .context(
                        "FP16 CE-tape rows16-cluster4 cross-row projection kernel is unavailable",
                    )?,
                &self.cross_entropy_row_stats_tile_partials_rows16,
                16,
            )),
            _ => bail!(
                "LM execution arm {} is not a cross-row CE-tape arm",
                arm.label()
            ),
        }
    }

    fn fp16_lm_weight_grad_kernel(
        &self,
        topology: HierarchosLmWeightGradTopology,
    ) -> Result<&vulkan::ComputeKernel> {
        if self.native_fp16_lm_backward_compute {
            return match topology {
                HierarchosLmWeightGradTopology::VocabRows4 => self
                    .cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows4
                    .as_ref()
                    .context("native-FP16-compute LM dW vocab4 topology is unavailable"),
                HierarchosLmWeightGradTopology::VocabRows8 => self
                    .cross_entropy_linear_weight_grad_streaming_fp16_native_compute
                    .as_ref()
                    .context("native-FP16-compute LM dW vocab8 topology is unavailable"),
                HierarchosLmWeightGradTopology::VocabRows16 => self
                    .cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows16
                    .as_ref()
                    .context("native-FP16-compute LM dW vocab16 topology is unavailable"),
            };
        }
        match topology {
            HierarchosLmWeightGradTopology::VocabRows4 => self
                .cross_entropy_linear_weight_grad_streaming_fp16_packed_rows4
                .as_ref()
                .context("FP16 LM dW vocab4 topology is unavailable"),
            HierarchosLmWeightGradTopology::VocabRows8 => self
                .cross_entropy_linear_weight_grad_streaming_fp16_packed
                .as_ref()
                .context("FP16 LM dW vocab8 topology is unavailable"),
            HierarchosLmWeightGradTopology::VocabRows16 => self
                .cross_entropy_linear_weight_grad_streaming_fp16_packed_rows16
                .as_ref()
                .context("FP16 LM dW vocab16 topology is unavailable"),
        }
    }

    fn fp16_lm_tape_weight_grad_kernel(
        &self,
        topology: HierarchosLmWeightGradTopology,
    ) -> Result<&vulkan::ComputeKernel> {
        match topology {
            HierarchosLmWeightGradTopology::VocabRows4 => self
                .cross_entropy_linear_weight_grad_tape_fp16_packed_rows4
                .as_ref()
                .context("FP16 LM direct-logit dW vocab4 topology is unavailable"),
            HierarchosLmWeightGradTopology::VocabRows8 => self
                .cross_entropy_linear_weight_grad_tape_fp16_packed
                .as_ref()
                .context("FP16 LM direct-logit dW vocab8 topology is unavailable"),
            HierarchosLmWeightGradTopology::VocabRows16 => self
                .cross_entropy_linear_weight_grad_tape_fp16_packed_rows16
                .as_ref()
                .context("FP16 LM direct-logit dW vocab16 topology is unavailable"),
        }
    }

    /// Select the fastest numerically equivalent packed-FP16 LM consumer for
    /// this device/geometry and persist the result. FP32 remains the outer
    /// precision-policy arm; this inner race never changes the quantized
    /// execution weights seen by the PyTorch parity oracle.
    pub(crate) fn configure_fp16_lm_execution_arm(&mut self) -> Result<()> {
        let fp16_weight = self
            .lm_head
            .fp16_parameter_storage_mirror()?
            .context("LM FP16 execution autotune requires an installed packed parameter mirror")?;
        let rows = self.max_rows;
        if self
            .cross_entropy_linear_adjoints_tape_fp16_packed_fused
            .is_some()
        {
            let profile_partial_len = self
                .vocab_size
                .div_ceil(CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_VOCAB_TILE)
                .checked_mul(rows)
                .and_then(|count| count.checked_mul(self.context_dim))
                .context("LM autotune dX partial count overflow")?;
            let needs_profile_scratch = self
                .ce_input_grad_partials
                .as_ref()
                .is_none_or(|buffer| buffer.f32_capacity() < profile_partial_len);
            if needs_profile_scratch {
                self.ce_input_grad_partials =
                    Some(GpuBuffer::zeros_f32(&self.device, profile_partial_len)?);
            }
        }
        let hidden_len = rows
            .checked_mul(self.context_dim)
            .context("LM autotune hidden length overflow")?;
        let synthetic_hidden = (0..hidden_len)
            .map(|index| {
                let centered = (index % 31) as f32 - 15.0;
                centered * (1.0 / 32.0)
            })
            .collect::<Vec<_>>();
        let synthetic_targets = (0..rows)
            .map(|row| ((row.wrapping_mul(7919).wrapping_add(17)) % self.vocab_size) as u32)
            .collect::<Vec<_>>();

        // Populate only trainer-owned scratch. This runs during graph
        // construction, before any caller-owned activation or gradient window
        // exists, so the one-time profile cannot perturb a training step.
        let mut setup = vulkan::ComputeBatch::new(&self.device)?;
        setup.upload_f32(&self.input_hidden, &synthetic_hidden)?;
        setup.upload_u32(&self.targets, &synthetic_targets)?;
        let linear_push = self.record_norm_head_stats(&mut setup, &self.input_hidden, rows)?;
        setup.submit()?;

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let native_fp16_candidate = self
            .cross_entropy_linear_input_grad_streaming_fp16_native
            .is_some();
        let geometry = lm_execution::LmExecutionAutotuneGeometry {
            device_name: self.device.name(),
            subgroup_size,
            context_dim: self.context_dim,
            vocab_size: self.vocab_size,
            rows,
            native_fp16_candidate,
            max_compute_shared_memory_bytes: self.device.max_compute_shared_memory_bytes(),
            ce_tape_candidate: self
                .cross_entropy_linear_row_stats_streaming_fp16_packed_tape
                .is_some()
                && self.device.supports_compute_work_group_size_x(128),
            ce_tape_rows8_candidate: self
                .cross_entropy_linear_logit_tape_fp16_packed_rows8
                .is_some()
                && self.device.supports_compute_work_group_size_x(128),
            ce_tape_rows16_candidate: self
                .cross_entropy_linear_logit_tape_fp16_packed_rows16
                .is_some()
                && (self.device.max_compute_shared_memory_bytes() < 16_384
                    || self
                        .cross_entropy_linear_logit_tape_fp16_packed_rows16_dot4
                        .is_some())
                && self.device.supports_compute_work_group_size_x(128),
            ce_tape_rows16_fused_adjoints_candidate: self
                .cross_entropy_linear_logit_tape_fp16_packed_rows16
                .is_some()
                && self
                    .cross_entropy_linear_adjoints_tape_fp16_packed_fused
                    .is_some()
                && self.cross_entropy_linear_input_grad_tile_reduce.is_some()
                && self.ce_input_grad_partials.is_some()
                && self.device.supports_compute_work_group_size_x(128),
            fused_adjoints_private_hidden_candidate: self
                .cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden
                .is_some(),
            fused_adjoints_private_hidden_tile256_candidate: self
                .cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256
                .is_some()
                && self
                    .cross_entropy_linear_input_grad_tile_reduce_tile256
                    .is_some(),
            ce_tape_rows16_cluster4_candidate: self
                .cross_entropy_linear_logit_tape_fp16_packed_rows16_cluster4
                .is_some(),
            dw_vocab4_candidate: self
                .cross_entropy_linear_weight_grad_streaming_fp16_packed_rows4
                .is_some(),
            dw_vocab8_candidate: self
                .cross_entropy_linear_weight_grad_streaming_fp16_packed
                .is_some(),
            dw_vocab16_candidate: self
                .cross_entropy_linear_weight_grad_streaming_fp16_packed_rows16
                .is_some(),
            kernel_signature: lm_execution_autotune_kernel_signature(),
        };
        let selected = lm_execution::choose_fp16_backward_plan(geometry, |plan| {
            self.time_fp16_lm_backward_plan_ms(&fp16_weight, rows, linear_push, plan)
        })?;
        self.lm_execution_arm = selected.input_grad_arm;
        self.lm_weight_grad_topology = selected.weight_grad_topology;
        self.lm_fused_adjoint_topology = selected.fused_adjoint_topology;
        if selected.input_grad_arm.fuses_ce_adjoints() {
            let selected_partial_len = self
                .vocab_size
                .div_ceil(selected.fused_adjoint_topology.vocab_tile())
                .checked_mul(rows)
                .and_then(|count| count.checked_mul(self.context_dim))
                .context("selected LM fused-adjoint dX partial count overflow")?;
            if self
                .ce_input_grad_partials
                .as_ref()
                .is_none_or(|buffer| buffer.f32_capacity() != selected_partial_len)
            {
                self.ce_input_grad_partials =
                    Some(GpuBuffer::zeros_f32(&self.device, selected_partial_len)?);
            }
        } else {
            self.ce_input_grad_partials = None;
        }
        Ok(())
    }

    /// Opt into the first numerically distinct native-FP16 backward tranche.
    /// CE/log-sum-exp remains FP32, but its scaled adjoint is rounded to FP16
    /// before the LM W^T and dW products. Products execute in native FP16 and
    /// accumulate into the existing FP32 gradient buffers, preserving the
    /// canonical master/optimizer/checkpoint ABI and GradScaler unscale path.
    pub(crate) fn enable_native_fp16_lm_backward_compute(
        &mut self,
        native_input_grad_compute: bool,
    ) -> Result<()> {
        self.lm_head
            .fp16_parameter_storage_mirror()?
            .context("native-FP16 LM backward compute requires an installed FP16 mirror")?;
        let native_input_grad_compute = native_input_grad_compute
            && std::env::var_os(HIERARCHOS_VULKAN_DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV).is_none();
        if !native_input_grad_compute {
            self.cross_entropy_linear_input_grad_streaming_fp16_native
                .as_ref()
                .context("device cannot create the FP32-compute/FP16-storage LM input adjoint")?;
        } else {
            self.cross_entropy_linear_input_grad_streaming_fp16_native_compute
                .as_ref()
                .context("device cannot create the native-FP16-compute LM input adjoint")?;
        }
        match self.lm_weight_grad_topology {
            HierarchosLmWeightGradTopology::VocabRows4 => self
                .cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows4
                .as_ref()
                .context("device cannot create native-FP16-compute LM dW vocab4")?,
            HierarchosLmWeightGradTopology::VocabRows8 => self
                .cross_entropy_linear_weight_grad_streaming_fp16_native_compute
                .as_ref()
                .context("device cannot create native-FP16-compute LM dW vocab8")?,
            HierarchosLmWeightGradTopology::VocabRows16 => self
                .cross_entropy_linear_weight_grad_streaming_fp16_native_compute_rows16
                .as_ref()
                .context("device cannot create native-FP16-compute LM dW vocab16")?,
        };
        self.native_fp16_lm_backward_compute = true;
        self.native_fp16_lm_input_grad_compute = native_input_grad_compute;
        // The first compute implementation is the portable 32-pair native
        // storage topology. Keep the numerically distinct arm out of the
        // existing equivalence-only LM autotuner until it has its own parity
        // and performance evidence.
        self.lm_execution_arm = HierarchosLmExecutionArm::Fp16Native;
        self.lm_fused_adjoint_topology =
            lm_execution::HierarchosLmFusedAdjointTopology::SharedHidden;
        self.ce_input_grad_partials = None;
        Ok(())
    }

    /// Extend the native-FP16 backward contract through `out_norm` without
    /// changing the FP32 LayerNorm statistics, reductions, destination
    /// gradients, optimizer moments, or checkpoint parameters.
    pub(crate) fn enable_native_fp16_out_norm_backward_compute(&mut self) -> Result<()> {
        self.layer_norm_input_grad_fp16_native_compute
            .as_ref()
            .context("device cannot create native-FP16 out_norm input backward")?;
        self.layer_norm_param_grad_fp16_native_compute
            .as_ref()
            .context("device cannot create native-FP16 out_norm parameter backward")?;
        self.native_fp16_out_norm_backward_compute = true;
        Ok(())
    }

    fn fp16_fused_adjoint_kernel(
        &self,
        topology: lm_execution::HierarchosLmFusedAdjointTopology,
    ) -> Result<&vulkan::ComputeKernel> {
        match topology {
            lm_execution::HierarchosLmFusedAdjointTopology::SharedHidden => self
                .cross_entropy_linear_adjoints_tape_fp16_packed_fused
                .as_ref()
                .context("FP16 rows16 fused-adjoint shared-hidden kernel is unavailable"),
            lm_execution::HierarchosLmFusedAdjointTopology::PrivateHidden => self
                .cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden
                .as_ref()
                .context("FP16 rows16 fused-adjoint private-hidden kernel is unavailable"),
            lm_execution::HierarchosLmFusedAdjointTopology::PrivateHiddenTile256 => self
                .cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256
                .as_ref()
                .context("FP16 rows16 fused-adjoint private-hidden tile256 kernel is unavailable"),
        }
    }

    fn fp16_fused_adjoint_reduce_kernel(
        &self,
        topology: lm_execution::HierarchosLmFusedAdjointTopology,
    ) -> Result<&vulkan::ComputeKernel> {
        match topology {
            lm_execution::HierarchosLmFusedAdjointTopology::SharedHidden
            | lm_execution::HierarchosLmFusedAdjointTopology::PrivateHidden => self
                .cross_entropy_linear_input_grad_tile_reduce
                .as_ref()
                .context("FP16 rows16 fused-adjoint dX reduction kernel is unavailable"),
            lm_execution::HierarchosLmFusedAdjointTopology::PrivateHiddenTile256 => self
                .cross_entropy_linear_input_grad_tile_reduce_tile256
                .as_ref()
                .context("FP16 rows16 fused-adjoint tile256 dX reduction kernel is unavailable"),
        }
    }

    fn record_fp16_lm_profile(
        &self,
        batch: &mut vulkan::ComputeBatch,
        fp16_weight: &mixed_precision::VulkanParameterStorageMirror,
        rows: usize,
        linear_push: LinearPush,
        plan: lm_execution::HierarchosLmBackwardPlan,
    ) -> Result<()> {
        let arm = plan.input_grad_arm;
        if !arm.uses_fp16_weights() {
            bail!("LM FP16 profiler cannot execute arm {}", arm.label());
        }
        let weight_grad_push = LmWeightGradPush {
            rows: linear_push.rows,
            input_dim: linear_push.input_dim,
            output_dim: linear_push.output_dim,
            accumulate: 0,
            z_loss_weight: 0.0,
            activation_clamp: linear_push.activation_clamp,
        };
        if arm == HierarchosLmExecutionArm::Fp16CeTape {
            self.cross_entropy_linear_row_stats_streaming_fp16_packed_tape
                .as_ref()
                .context("FP16 CE-tape row-stats kernel is unavailable")?
                .record_dispatch(
                    batch,
                    &[
                        &self.input_hidden,
                        &self.norm_weight,
                        &self.norm_bias,
                        &self.norm_mean,
                        &self.norm_rstd,
                        fp16_weight.packed_storage(),
                        &self.targets,
                        &self.ce_row_stats,
                        &self.ce_grad_tape,
                    ],
                    bytemuck::bytes_of(&linear_push),
                    [rows as u32, 1, 1],
                )?;
            self.cross_entropy_linear_input_grad_tape_fp16_packed
                .record_dispatch(
                    batch,
                    &[
                        &self.ce_grad_tape,
                        &self.ce_row_stats,
                        fp16_weight.packed_storage(),
                        &self.grad_norm_hidden,
                    ],
                    bytemuck::bytes_of(&linear_push),
                    [div_ceil_u32(self.context_dim.div_ceil(2), 32), 1, 1],
                )?;
        } else if matches!(
            arm,
            HierarchosLmExecutionArm::Fp16CeTapeRows8
                | HierarchosLmExecutionArm::Fp16CeTapeRows16
                | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
                | HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints
                | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints
                | HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints
        ) {
            let (projection_kernel, stats_kernel, vocab_tile) =
                self.fp16_ce_cross_row_kernels(arm)?;
            projection_kernel.record_dispatch(
                batch,
                &[
                    &self.input_hidden,
                    &self.norm_weight,
                    &self.norm_bias,
                    &self.norm_mean,
                    &self.norm_rstd,
                    fp16_weight.packed_storage(),
                    &self.targets,
                    &self.ce_grad_tape,
                    &self.ce_tile_stats,
                ],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(self.vocab_size, vocab_tile), 1, 1],
            )?;
            stats_kernel.record_dispatch(
                batch,
                &[&self.ce_tile_stats, &self.targets, &self.ce_row_stats],
                bytemuck::bytes_of(&linear_push),
                [rows as u32, 1, 1],
            )?;
            if arm.fuses_ce_adjoints() {
                let partials = self
                    .ce_input_grad_partials
                    .as_ref()
                    .context("FP16 rows16 fused-adjoint dX partial scratch is unavailable")?;
                self.fp16_fused_adjoint_kernel(plan.fused_adjoint_topology)?
                    .record_dispatch(
                        batch,
                        &[
                            &self.input_hidden,
                            &self.norm_weight,
                            &self.norm_bias,
                            &self.norm_mean,
                            &self.norm_rstd,
                            &self.ce_grad_tape,
                            &self.ce_row_stats,
                            fp16_weight.packed_storage(),
                            &self.grad_lm_weight,
                            partials,
                        ],
                        bytemuck::bytes_of(&weight_grad_push),
                        [
                            div_ceil_u32(self.vocab_size, plan.fused_adjoint_topology.vocab_tile()),
                            1,
                            1,
                        ],
                    )?;
                self.fp16_fused_adjoint_reduce_kernel(plan.fused_adjoint_topology)?
                    .record_dispatch(
                        batch,
                        &[partials, &self.grad_norm_hidden],
                        bytemuck::bytes_of(&linear_push),
                        [div_ceil_u32(rows * self.context_dim, 64), 1, 1],
                    )?;
            } else {
                self.cross_entropy_linear_input_grad_tape_fp16_packed
                    .record_dispatch(
                        batch,
                        &[
                            &self.ce_grad_tape,
                            &self.ce_row_stats,
                            fp16_weight.packed_storage(),
                            &self.grad_norm_hidden,
                        ],
                        bytemuck::bytes_of(&linear_push),
                        [div_ceil_u32(self.context_dim.div_ceil(2), 32), 1, 1],
                    )?;
            }
        } else {
            self.cross_entropy_linear_row_stats_streaming_fp16_packed
                .record_dispatch(
                    batch,
                    &[
                        &self.input_hidden,
                        &self.norm_weight,
                        &self.norm_bias,
                        &self.norm_mean,
                        &self.norm_rstd,
                        fp16_weight.packed_storage(),
                        &self.targets,
                        &self.ce_row_stats,
                    ],
                    bytemuck::bytes_of(&linear_push),
                    [rows as u32, 1, 1],
                )?;
            let input_grad_kernel = self.fp16_lm_input_grad_kernel(arm)?;
            input_grad_kernel.record_dispatch(
                batch,
                &[
                    &self.input_hidden,
                    &self.norm_weight,
                    &self.norm_bias,
                    &self.norm_mean,
                    &self.norm_rstd,
                    fp16_weight.packed_storage(),
                    &self.ce_row_stats,
                    &self.grad_norm_hidden,
                    &self.ce_grad_tape,
                ],
                bytemuck::bytes_of(&linear_push),
                [rows as u32, 1, 1],
            )?;
        }
        if !arm.fuses_ce_adjoints() {
            let dweight_groups = [
                div_ceil_u32(
                    self.vocab_size,
                    plan.weight_grad_topology.vocab_rows_per_group() as usize,
                ),
                1,
                1,
            ];
            if matches!(
                arm,
                HierarchosLmExecutionArm::Fp16CeTape
                    | HierarchosLmExecutionArm::Fp16CeTapeRows8
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
            ) {
                self.fp16_lm_tape_weight_grad_kernel(plan.weight_grad_topology)?
                    .record_dispatch(
                        batch,
                        &[
                            &self.input_hidden,
                            &self.norm_weight,
                            &self.norm_bias,
                            &self.norm_mean,
                            &self.norm_rstd,
                            &self.ce_grad_tape,
                            &self.ce_row_stats,
                            &self.grad_lm_weight,
                        ],
                        bytemuck::bytes_of(&weight_grad_push),
                        dweight_groups,
                    )?;
            } else {
                self.fp16_lm_weight_grad_kernel(plan.weight_grad_topology)?
                    .record_dispatch(
                        batch,
                        &[
                            &self.input_hidden,
                            &self.norm_weight,
                            &self.norm_bias,
                            &self.norm_mean,
                            &self.norm_rstd,
                            &self.ce_grad_tape,
                            &self.grad_lm_weight,
                        ],
                        bytemuck::bytes_of(&weight_grad_push),
                        dweight_groups,
                    )?;
            }
        }
        Ok(())
    }

    fn time_fp16_lm_backward_plan_ms(
        &self,
        fp16_weight: &mixed_precision::VulkanParameterStorageMirror,
        rows: usize,
        linear_push: LinearPush,
        plan: lm_execution::HierarchosLmBackwardPlan,
    ) -> Result<f64> {
        let matrix_len = self
            .context_dim
            .checked_mul(self.vocab_size)
            .context("LM autotune matrix size overflow")?;
        let repetitions = if matrix_len >= 1_000_000 { 1 } else { 4 };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..repetitions {
            self.record_fp16_lm_profile(&mut commands, fp16_weight, rows, linear_push, plan)?;
        }
        let started = std::time::Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    /// Stream `out_norm` reconstruction and the vocabulary dimension through
    /// LM projection, CE statistics, and both LM-head adjoints without
    /// allocating `[rows, context_dim]` normalized activations. Packed-FP16
    /// execution keeps one bounded reusable FP32 tape. The legacy tape arm
    /// captures logits in its row-major CE forward; the cross-row tape arms
    /// reuse a resident vocabulary tile of W across Hierarchos rows and emit
    /// compact max/scaled-sum/target-logit partials beside the tape. Row
    /// statistics reduce those partials; W^T and dW then derive each CE
    /// adjoint directly from the preserved logit plus four-value row stats,
    /// avoiding a global logits-to-grad tape rewrite.
    fn record_streaming_lm_loss_backward(
        &self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        linear_push: LinearPush,
    ) -> Result<()> {
        self.record_streaming_lm_loss_backward_with_dense_target(
            batch,
            pre_norm_hidden,
            rows,
            linear_push,
            LmWeightGradWriteMode::ScratchOverwrite,
            0.0,
            false,
        )
    }

    /// Variant for the full-model graph that can target the persistent shared
    /// tied-LM accumulator directly. On the first contribution in an
    /// accumulation window dW overwrites that freshly reset buffer; subsequent
    /// microbatches add in-place. The scratch mode remains available as a
    /// profiling/debug fallback and still performs the historical
    /// `gradient_accumulate` sweep.
    fn record_streaming_lm_loss_backward_with_dense_target(
        &self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        linear_push: LinearPush,
        weight_grad_write_mode: LmWeightGradWriteMode,
        z_loss_weight: f32,
        force_fp32_source_adjoints: bool,
    ) -> Result<()> {
        self.record_streaming_lm_loss_with_dense_target(
            batch,
            pre_norm_hidden,
            rows,
            linear_push,
            weight_grad_write_mode,
            z_loss_weight,
            force_fp32_source_adjoints,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_streaming_lm_loss_with_dense_target(
        &self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        linear_push: LinearPush,
        weight_grad_write_mode: LmWeightGradWriteMode,
        z_loss_weight: f32,
        force_fp32_source_adjoints: bool,
        record_backward: bool,
    ) -> Result<()> {
        if !z_loss_weight.is_finite() || z_loss_weight < 0.0 {
            bail!("LM z-loss weight must be finite and non-negative");
        }
        let loss_push = LinearPush {
            z_loss_weight,
            ..linear_push
        };
        let dense_lm_grad = if weight_grad_write_mode.uses_shared_buffer() {
            self.lm_head.gradient_buffer()
        } else {
            &self.grad_lm_weight
        };
        let weight_grad_push = LmWeightGradPush {
            rows: linear_push.rows,
            input_dim: linear_push.input_dim,
            output_dim: linear_push.output_dim,
            accumulate: u32::from(weight_grad_write_mode.accumulates_existing()),
            z_loss_weight,
            activation_clamp: linear_push.activation_clamp,
        };
        let fp16_lm_weight = if self.lm_execution_arm.uses_fp16_weights() {
            Some(
                self.lm_head
                    .fp16_parameter_storage_mirror()?
                    .context("selected FP16 LM execution arm has no installed parameter mirror")?,
            )
        } else {
            None
        };
        match self.lm_execution_arm {
            HierarchosLmExecutionArm::Fp32 => {
                self.cross_entropy_linear_row_stats_streaming
                    .record_dispatch(
                        batch,
                        &[
                            pre_norm_hidden,
                            &self.norm_weight,
                            &self.norm_bias,
                            &self.norm_mean,
                            &self.norm_rstd,
                            self.lm_head.weight_buffer(),
                            &self.targets,
                            &self.ce_row_stats,
                        ],
                        bytemuck::bytes_of(&linear_push),
                        [rows as u32, 1, 1],
                    )?;
            }
            HierarchosLmExecutionArm::Fp16Packed
            | HierarchosLmExecutionArm::Fp16Native
            | HierarchosLmExecutionArm::Fp16NativeReuse64
            | HierarchosLmExecutionArm::Fp16NativeReuse128
            | HierarchosLmExecutionArm::Fp16NativeReuse224 => {
                let fp16_weight = fp16_lm_weight
                    .as_ref()
                    .expect("FP16 execution arm validated packed mirror");
                self.cross_entropy_linear_row_stats_streaming_fp16_packed
                    .record_dispatch(
                        batch,
                        &[
                            pre_norm_hidden,
                            &self.norm_weight,
                            &self.norm_bias,
                            &self.norm_mean,
                            &self.norm_rstd,
                            fp16_weight.packed_storage(),
                            &self.targets,
                            &self.ce_row_stats,
                        ],
                        bytemuck::bytes_of(&linear_push),
                        [rows as u32, 1, 1],
                    )?;
            }
            HierarchosLmExecutionArm::Fp16CeTape => {
                let fp16_weight = fp16_lm_weight
                    .as_ref()
                    .expect("FP16 CE-tape arm validated packed mirror");
                self.cross_entropy_linear_row_stats_streaming_fp16_packed_tape
                    .as_ref()
                    .context("FP16 CE-tape row-stats kernel is unavailable")?
                    .record_dispatch(
                        batch,
                        &[
                            pre_norm_hidden,
                            &self.norm_weight,
                            &self.norm_bias,
                            &self.norm_mean,
                            &self.norm_rstd,
                            fp16_weight.packed_storage(),
                            &self.targets,
                            &self.ce_row_stats,
                            &self.ce_grad_tape,
                        ],
                        bytemuck::bytes_of(&linear_push),
                        [rows as u32, 1, 1],
                    )?;
            }
            HierarchosLmExecutionArm::Fp16CeTapeRows8
            | HierarchosLmExecutionArm::Fp16CeTapeRows16
            | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
            | HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints
            | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints
            | HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints => {
                let fp16_weight = fp16_lm_weight
                    .as_ref()
                    .expect("FP16 cross-row CE-tape arm validated packed mirror");
                let (projection_kernel, stats_kernel, vocab_tile) =
                    self.fp16_ce_cross_row_kernels(self.lm_execution_arm)?;
                projection_kernel.record_dispatch(
                    batch,
                    &[
                        pre_norm_hidden,
                        &self.norm_weight,
                        &self.norm_bias,
                        &self.norm_mean,
                        &self.norm_rstd,
                        fp16_weight.packed_storage(),
                        &self.targets,
                        &self.ce_grad_tape,
                        &self.ce_tile_stats,
                    ],
                    bytemuck::bytes_of(&loss_push),
                    [div_ceil_u32(self.vocab_size, vocab_tile), 1, 1],
                )?;
                stats_kernel.record_dispatch(
                    batch,
                    &[&self.ce_tile_stats, &self.targets, &self.ce_row_stats],
                    bytemuck::bytes_of(&linear_push),
                    [rows as u32, 1, 1],
                )?;
            }
        }

        let xent_push = CrossEntropyPush {
            rows: rows as u32,
            vocab_size: self.vocab_size as u32,
            z_loss_weight,
        };
        self.cross_entropy_row_loss_extract.record_dispatch(
            batch,
            &[&self.ce_row_stats, &self.row_loss],
            bytemuck::bytes_of(&xent_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;

        // Scalar backward caps need the exact forward CE+z-loss value before
        // any component adjoint is recorded. The cap-only preflight reuses the
        // same streaming projection/statistics kernels, then stops here: no LM,
        // out_norm, recurrent, or optimizer gradient buffer is touched.
        if !record_backward {
            return Ok(());
        }

        if let Some(fp16_weight) = fp16_lm_weight.as_ref() {
            if self.lm_execution_arm.fuses_ce_adjoints() {
                // A vocabulary-tile owner consumes each taped logit once,
                // sweeps each W element once, emits dW directly, and writes a
                // deterministic dX partial. Tile width is an autotuned fused
                // topology coordinate; the follow-up reduction consumes the
                // matching partial layout without changing checkpoint/logit
                // representation.
                let partials = self
                    .ce_input_grad_partials
                    .as_ref()
                    .context("FP16 rows16 fused-adjoint dX partial scratch is unavailable")?;
                self.fp16_fused_adjoint_kernel(self.lm_fused_adjoint_topology)?
                    .record_dispatch(
                        batch,
                        &[
                            pre_norm_hidden,
                            &self.norm_weight,
                            &self.norm_bias,
                            &self.norm_mean,
                            &self.norm_rstd,
                            &self.ce_grad_tape,
                            &self.ce_row_stats,
                            fp16_weight.packed_storage(),
                            dense_lm_grad,
                            partials,
                        ],
                        bytemuck::bytes_of(&weight_grad_push),
                        [
                            div_ceil_u32(
                                self.vocab_size,
                                self.lm_fused_adjoint_topology.vocab_tile(),
                            ),
                            1,
                            1,
                        ],
                    )?;
                self.fp16_fused_adjoint_reduce_kernel(self.lm_fused_adjoint_topology)?
                    .record_dispatch(
                        batch,
                        &[partials, &self.grad_norm_hidden],
                        bytemuck::bytes_of(&linear_push),
                        [div_ceil_u32(rows * self.context_dim, 64), 1, 1],
                    )?;
            } else if matches!(
                self.lm_execution_arm,
                HierarchosLmExecutionArm::Fp16CeTape
                    | HierarchosLmExecutionArm::Fp16CeTapeRows8
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
            ) {
                // Forward already produced these logits. W^T consumes each CE
                // adjoint directly from the logit plus compact row stats, so
                // there is no full-tape logits->grad rewrite between forward
                // and backward. Cross-row modes still reduce compact
                // projection-side tile partials instead of rereading the tape.
                self.cross_entropy_linear_input_grad_tape_fp16_packed
                    .record_dispatch(
                        batch,
                        &[
                            &self.ce_grad_tape,
                            &self.ce_row_stats,
                            fp16_weight.packed_storage(),
                            &self.grad_norm_hidden,
                        ],
                        bytemuck::bytes_of(&linear_push),
                        [div_ceil_u32(self.context_dim.div_ceil(2), 32), 1, 1],
                    )?;
            } else {
                // The streaming input adjoint regenerates logits for W^T and
                // captures each final CE adjoint once for dW to consume.
                // GradScaler-style source domains can lift tiny, cancelling CE
                // adjoints into a range where an FP16 round-trip changes their
                // post-unscale sign. Keep the native-FP16 dW product active, but
                // route W^T through the already-qualified FP32-compute / FP16-
                // storage kernel for those domains.
                let input_grad_kernel = if force_fp32_source_adjoints
                    && self.native_fp16_lm_input_grad_compute_active()
                {
                    self.cross_entropy_linear_input_grad_streaming_fp16_native
                        .as_ref()
                        .context(
                            "source-scaled LM input adjoint requires the FP32-compute/FP16-storage kernel",
                        )?
                } else {
                    self.fp16_lm_input_grad_kernel(self.lm_execution_arm)?
                };
                input_grad_kernel.record_dispatch(
                    batch,
                    &[
                        pre_norm_hidden,
                        &self.norm_weight,
                        &self.norm_bias,
                        &self.norm_mean,
                        &self.norm_rstd,
                        fp16_weight.packed_storage(),
                        &self.ce_row_stats,
                        &self.grad_norm_hidden,
                        &self.ce_grad_tape,
                    ],
                    bytemuck::bytes_of(&loss_push),
                    [rows as u32, 1, 1],
                )?;
            }

            if !self.lm_execution_arm.fuses_ce_adjoints() {
                let dweight_groups = [
                    div_ceil_u32(
                        self.vocab_size,
                        self.lm_weight_grad_topology.vocab_rows_per_group() as usize,
                    ),
                    1,
                    1,
                ];
                if matches!(
                    self.lm_execution_arm,
                    HierarchosLmExecutionArm::Fp16CeTape
                        | HierarchosLmExecutionArm::Fp16CeTapeRows8
                        | HierarchosLmExecutionArm::Fp16CeTapeRows16
                        | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
                ) {
                    self.fp16_lm_tape_weight_grad_kernel(self.lm_weight_grad_topology)?
                        .record_dispatch(
                            batch,
                            &[
                                pre_norm_hidden,
                                &self.norm_weight,
                                &self.norm_bias,
                                &self.norm_mean,
                                &self.norm_rstd,
                                &self.ce_grad_tape,
                                &self.ce_row_stats,
                                dense_lm_grad,
                            ],
                            bytemuck::bytes_of(&weight_grad_push),
                            dweight_groups,
                        )?;
                } else {
                    self.fp16_lm_weight_grad_kernel(self.lm_weight_grad_topology)?
                        .record_dispatch(
                            batch,
                            &[
                                pre_norm_hidden,
                                &self.norm_weight,
                                &self.norm_bias,
                                &self.norm_mean,
                                &self.norm_rstd,
                                &self.ce_grad_tape,
                                dense_lm_grad,
                            ],
                            bytemuck::bytes_of(&weight_grad_push),
                            dweight_groups,
                        )?;
                }
            }
            if !weight_grad_write_mode.uses_shared_buffer() {
                self.lm_head
                    .record_accumulate_gradient(batch, &self.grad_lm_weight)?;
            }
        } else {
            self.cross_entropy_linear_weight_grad_streaming
                .record_dispatch(
                    batch,
                    &[
                        pre_norm_hidden,
                        &self.norm_weight,
                        &self.norm_bias,
                        &self.norm_mean,
                        &self.norm_rstd,
                        self.lm_head.weight_buffer(),
                        &self.ce_row_stats,
                        dense_lm_grad,
                    ],
                    bytemuck::bytes_of(&weight_grad_push),
                    [
                        div_ceil_u32(self.vocab_size, CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_VOCAB_TILE),
                        1,
                        1,
                    ],
                )?;
            if !weight_grad_write_mode.uses_shared_buffer() {
                self.lm_head
                    .record_accumulate_gradient(batch, &self.grad_lm_weight)?;
            }
            self.cross_entropy_linear_input_grad_streaming
                .record_dispatch(
                    batch,
                    &[
                        pre_norm_hidden,
                        &self.norm_weight,
                        &self.norm_bias,
                        &self.norm_mean,
                        &self.norm_rstd,
                        self.lm_head.weight_buffer(),
                        &self.ce_row_stats,
                        &self.grad_norm_hidden,
                    ],
                    bytemuck::bytes_of(&loss_push),
                    [rows as u32, 1, 1],
                )?;
        }
        Ok(())
    }

    /// Record the exact per-row CE + z-loss forward scalar without recording
    /// any adjoint kernel. This is used by native sequence-scalar backward caps:
    /// the forward probe is still entirely Vulkan, while Rust only evaluates
    /// the scalar `minimum(value, ceiling)` derivative gate between submissions.
    pub(crate) fn record_loss_forward_from_buffer_with_z_loss_into(
        &self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        z_loss_weight: f32,
        row_loss_readback: &GpuBuffer,
    ) -> Result<()> {
        if !z_loss_weight.is_finite() || z_loss_weight < 0.0 {
            bail!("output-loss z-loss weight must be finite and non-negative");
        }
        if rows == 0 || rows > self.max_rows {
            bail!(
                "output-loss rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        let hidden_len = rows
            .checked_mul(self.context_dim)
            .context("output-loss hidden length overflow")?;
        if pre_norm_hidden.f32_capacity() < hidden_len {
            bail!(
                "output-loss device input capacity {} is smaller than required {hidden_len}",
                pre_norm_hidden.f32_capacity()
            );
        }
        if row_loss_readback.f32_capacity() < rows {
            bail!(
                "output-loss forward readback capacity {} is smaller than required {rows}",
                row_loss_readback.f32_capacity()
            );
        }
        if targets.len() != rows {
            bail!(
                "target count {} does not match output-loss rows {rows}",
                targets.len()
            );
        }
        if let Some(&bad) = targets
            .iter()
            .find(|&&target| target as usize >= self.vocab_size)
        {
            bail!(
                "target token {bad} is outside vocabulary size {}",
                self.vocab_size
            );
        }

        batch.upload_u32(&self.targets, targets)?;
        // Forward CE statistics own fields 0..3. Field 4 is the backward
        // source coefficient, so keep it explicitly zero in a metrics-only
        // pass to make accidental adjoint consumption fail harmlessly.
        batch.upload_f32(&self.ce_row_stats, &vec![0.0f32; rows * 5])?;
        let linear_push = self.record_norm_head_stats(batch, pre_norm_hidden, rows)?;
        self.record_streaming_lm_loss_with_dense_target(
            batch,
            pre_norm_hidden,
            rows,
            linear_push,
            LmWeightGradWriteMode::ScratchOverwrite,
            z_loss_weight,
            false,
            false,
        )?;
        batch.readback_f32(&self.row_loss, row_loss_readback, rows)
    }

    /// Full-graph loss backward when the tied `lm_head.weight` gradient was
    /// freshly cleared by `SharedLmHeadTrainMode::BeginAccumulation` earlier in
    /// the same command stream. This preserves the ordinary backward math while
    /// letting the dense LM dW kernel write the canonical shared accumulator
    /// directly; recurrent/token sparse gradients may accumulate into it later.
    pub(crate) fn record_loss_backward_from_buffer_fresh_shared_lm(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        supervision_weights: Option<&[f32]>,
    ) -> Result<HierarchosOutNormBackwardTicket> {
        self.record_loss_backward_from_buffer_shared_lm(
            batch,
            pre_norm_hidden,
            rows,
            targets,
            supervision_weights,
            false,
        )
    }

    /// Full-graph loss backward targeting the long-lived tied-LM gradient
    /// accumulator. `accumulate_shared_lm_grad` is false for the first reverse
    /// in an accumulation window and true for every later reverse, including
    /// those recorded in later queue submissions.
    pub(crate) fn record_loss_backward_from_buffer_shared_lm(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        supervision_weights: Option<&[f32]>,
        accumulate_shared_lm_grad: bool,
    ) -> Result<HierarchosOutNormBackwardTicket> {
        self.record_loss_backward_from_buffer_shared_lm_with_z_loss(
            batch,
            pre_norm_hidden,
            rows,
            targets,
            supervision_weights,
            0.0,
            1.0,
            false,
            accumulate_shared_lm_grad,
        )
    }

    pub(crate) fn record_loss_backward_from_buffer_shared_lm_with_z_loss(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        supervision_weights: Option<&[f32]>,
        z_loss_weight: f32,
        loss_source_scale: f32,
        source_scaled_backward_domain: bool,
        accumulate_shared_lm_grad: bool,
    ) -> Result<HierarchosOutNormBackwardTicket> {
        self.record_loss_backward_from_buffer_shared_lm_with_z_loss_and_objective_scale(
            batch,
            pre_norm_hidden,
            rows,
            targets,
            supervision_weights,
            z_loss_weight,
            loss_source_scale,
            1.0,
            source_scaled_backward_domain,
            accumulate_shared_lm_grad,
        )
    }

    /// Variant of the native CE+z-loss backward source with an independent
    /// objective derivative gate. `objective_source_scale` is intentionally
    /// separate from GradScaler's global source domain: a sequence-level CE cap
    /// may suppress only the language objective while ponder, commitment, LTM,
    /// and recurrent future-state adjoints remain live.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_loss_backward_from_buffer_shared_lm_with_z_loss_and_objective_scale(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        supervision_weights: Option<&[f32]>,
        z_loss_weight: f32,
        loss_source_scale: f32,
        objective_source_scale: f32,
        source_scaled_backward_domain: bool,
        accumulate_shared_lm_grad: bool,
    ) -> Result<HierarchosOutNormBackwardTicket> {
        self.record_loss_backward_from_buffer_shared_lm_with_z_loss_impl(
            batch,
            pre_norm_hidden,
            rows,
            targets,
            supervision_weights,
            z_loss_weight,
            loss_source_scale,
            objective_source_scale,
            source_scaled_backward_domain,
            None,
            accumulate_shared_lm_grad,
        )
    }

    /// Device-resident GradScaler variant of the full-graph loss source. The
    /// host uploads only the unscaled supervision coefficient; immediately
    /// before the CE adjoint kernels consume it, Vulkan multiplies field 4 of
    /// every compact row record by the controller's current scale. Forward
    /// loss diagnostics (fields 0..3) stay unscaled.
    pub(crate) fn record_loss_backward_from_buffer_shared_lm_with_z_loss_device_scaled(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        supervision_weights: Option<&[f32]>,
        z_loss_weight: f32,
        scaler: &training_numerics::VulkanDynamicLossScaleController,
        accumulate_shared_lm_grad: bool,
    ) -> Result<HierarchosOutNormBackwardTicket> {
        self.record_loss_backward_from_buffer_shared_lm_with_z_loss_device_scaled_and_objective_scale(
            batch,
            pre_norm_hidden,
            rows,
            targets,
            supervision_weights,
            z_loss_weight,
            1.0,
            scaler,
            accumulate_shared_lm_grad,
        )
    }

    /// Device-resident GradScaler counterpart of the CE-only objective gate.
    /// The host gate is folded into the compact row coefficient first; Vulkan
    /// then multiplies that coefficient by the live device scaler immediately
    /// before CE adjoints are generated.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_loss_backward_from_buffer_shared_lm_with_z_loss_device_scaled_and_objective_scale(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        supervision_weights: Option<&[f32]>,
        z_loss_weight: f32,
        objective_source_scale: f32,
        scaler: &training_numerics::VulkanDynamicLossScaleController,
        accumulate_shared_lm_grad: bool,
    ) -> Result<HierarchosOutNormBackwardTicket> {
        self.record_loss_backward_from_buffer_shared_lm_with_z_loss_impl(
            batch,
            pre_norm_hidden,
            rows,
            targets,
            supervision_weights,
            z_loss_weight,
            1.0,
            objective_source_scale,
            true,
            Some(scaler),
            accumulate_shared_lm_grad,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_loss_backward_from_buffer_shared_lm_with_z_loss_impl(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        targets: &[u32],
        supervision_weights: Option<&[f32]>,
        z_loss_weight: f32,
        loss_source_scale: f32,
        objective_source_scale: f32,
        source_scaled_backward_domain: bool,
        device_scaler: Option<&training_numerics::VulkanDynamicLossScaleController>,
        accumulate_shared_lm_grad: bool,
    ) -> Result<HierarchosOutNormBackwardTicket> {
        if !z_loss_weight.is_finite() || z_loss_weight < 0.0 {
            bail!("output-loss z-loss weight must be finite and non-negative");
        }
        if !loss_source_scale.is_finite() || loss_source_scale <= 0.0 {
            bail!("output-loss source scale must be finite and positive");
        }
        if !objective_source_scale.is_finite() || objective_source_scale < 0.0 {
            bail!("output-loss objective source scale must be finite and non-negative");
        }
        if rows == 0 || rows > self.max_rows {
            bail!(
                "output-loss rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        let hidden_len = rows
            .checked_mul(self.context_dim)
            .context("output-loss hidden length overflow")?;
        if pre_norm_hidden.f32_capacity() < hidden_len {
            bail!(
                "output-loss device input capacity {} is smaller than required {hidden_len}",
                pre_norm_hidden.f32_capacity()
            );
        }
        if targets.len() != rows {
            bail!(
                "target count {} does not match output-loss rows {rows}",
                targets.len()
            );
        }
        if let Some(&bad) = targets
            .iter()
            .find(|&&target| target as usize >= self.vocab_size)
        {
            bail!(
                "target token {bad} is outside vocabulary size {}",
                self.vocab_size
            );
        }
        if let Some(weights) = supervision_weights {
            if weights.len() != rows {
                bail!(
                    "supervision-weight count {} does not match output-loss rows {rows}",
                    weights.len()
                );
            }
            if let Some((row, &bad)) = weights
                .iter()
                .enumerate()
                .find(|(_, weight)| !weight.is_finite() || **weight < 0.0)
            {
                bail!("supervision weight at row {row} must be finite and non-negative; got {bad}");
            }
        }
        batch.upload_u32(&self.targets, targets)?;

        // The compact CE record carries the exact source-gradient coefficient
        // beside max/inv-sum/target/loss. Forward overwrites only the first four
        // values. Legacy callers retain the historical per-batch mean; token
        // tape callers may instead provide unnormalized PyTorch supervision
        // mass (including zero for padding / ignored labels).
        let legacy_mean_weight = 1.0 / rows as f32;
        let mut ce_row_stats = vec![0.0f32; rows * 5];
        for row in 0..rows {
            ce_row_stats[row * 5 + 4] = supervision_weights
                .map(|weights| weights[row])
                .unwrap_or(legacy_mean_weight)
                * loss_source_scale
                * objective_source_scale;
        }
        batch.upload_f32(&self.ce_row_stats, &ce_row_stats)?;

        let linear_push = self.record_norm_head_stats(batch, pre_norm_hidden, rows)?;
        if let Some(scaler) = device_scaler {
            scaler.record_scale_source_by_current_scale_strided(
                batch,
                &self.ce_row_stats,
                rows,
                5,
                4,
            )?;
        }
        let weight_grad_write_mode =
            if std::env::var_os(HIERARCHOS_VULKAN_LM_FORCE_DENSE_GRAD_STAGING_ENV).is_some() {
                LmWeightGradWriteMode::ScratchOverwrite
            } else if accumulate_shared_lm_grad {
                LmWeightGradWriteMode::SharedAccumulate
            } else {
                LmWeightGradWriteMode::SharedOverwrite
            };
        self.record_streaming_lm_loss_backward_with_dense_target(
            batch,
            pre_norm_hidden,
            rows,
            linear_push,
            weight_grad_write_mode,
            z_loss_weight,
            source_scaled_backward_domain,
        )?;
        self.record_out_norm_backward(batch, pre_norm_hidden, rows, source_scaled_backward_domain)?;

        Ok(HierarchosOutNormBackwardTicket { rows })
    }

    fn record_out_norm_backward(
        &self,
        batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &GpuBuffer,
        rows: usize,
        force_fp32_backward: bool,
    ) -> Result<()> {
        // Every LM execution arm writes the adjoint with respect to the
        // *clamped* affine out_norm value into this one canonical buffer.
        // Reconstruct only the scalar affine value needed by the clamp
        // Jacobian and mask that buffer in place before either LayerNorm
        // backward implementation consumes it. This keeps FP32, packed-FP16,
        // CE-tape, and fused-adjoint paths on one safety boundary without
        // materializing a [rows, context_dim] out_norm activation.
        let clamp_push = LayerNormAffineClampBackwardPush {
            rows: rows as u32,
            dim: self.context_dim as u32,
            max_abs: self.activation_clamp,
        };
        self.layer_norm_affine_clamp_backward_inplace
            .record_dispatch(
                batch,
                &[
                    pre_norm_hidden,
                    &self.norm_weight,
                    &self.norm_bias,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.grad_norm_hidden,
                ],
                bytemuck::bytes_of(&clamp_push),
                [div_ceil_u32(rows * self.context_dim, 256), 1, 1],
            )?;

        let push = LayerNormBackwardPush {
            rows: rows as u32,
            dim: self.context_dim as u32,
        };
        let use_native_fp16_backward =
            self.native_fp16_out_norm_backward_compute && !force_fp32_backward;
        let input_grad = if use_native_fp16_backward {
            self.layer_norm_input_grad_fp16_native_compute
                .as_ref()
                .context("native-FP16 out_norm input backward was enabled without a kernel")?
        } else {
            &self.layer_norm_input_grad
        };
        let param_grad = if use_native_fp16_backward {
            self.layer_norm_param_grad_fp16_native_compute
                .as_ref()
                .context("native-FP16 out_norm parameter backward was enabled without a kernel")?
        } else {
            &self.layer_norm_param_grad
        };
        input_grad.record_dispatch(
            batch,
            &[
                &self.grad_norm_hidden,
                pre_norm_hidden,
                &self.norm_weight,
                &self.norm_mean,
                &self.norm_rstd,
                &self.grad_input_hidden,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        param_grad.record_dispatch(
            batch,
            &[
                &self.grad_norm_hidden,
                pre_norm_hidden,
                &self.norm_mean,
                &self.norm_rstd,
                &self.grad_norm_weight,
                &self.grad_norm_bias,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(self.context_dim, 256), 1, 1],
        )?;
        Ok(())
    }

    /// Finalize a split output backward pass after every tied-LM consumer has
    /// contributed its gradient. This is the only phase that advances the LM
    /// and out_norm AdamW states.
    pub(crate) fn record_optimizer_step_after_backward(
        &mut self,
        batch: &mut vulkan::ComputeBatch,
        backward: HierarchosOutNormBackwardTicket,
        hyper: AdamWHyperParams,
    ) -> Result<HierarchosOutNormRecordedStep> {
        hyper.validate()?;
        let next_step = self.step.checked_add(1).context("AdamW step overflow")?;
        let lm_step = self.lm_head.record_step(batch, hyper)?;
        let norm_adam_push = AdamWPush {
            len: self.context_dim as u32,
            step: next_step,
            lr: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay: 0.0,
        };
        self.adamw.record_dispatch(
            batch,
            &[
                &self.norm_weight,
                &self.grad_norm_weight,
                &self.norm_weight_exp_avg,
                &self.norm_weight_exp_avg_sq,
            ],
            bytemuck::bytes_of(&norm_adam_push),
            [div_ceil_u32(self.context_dim, 256), 1, 1],
        )?;
        self.adamw.record_dispatch(
            batch,
            &[
                &self.norm_bias,
                &self.grad_norm_bias,
                &self.norm_bias_exp_avg,
                &self.norm_bias_exp_avg_sq,
            ],
            bytemuck::bytes_of(&norm_adam_push),
            [div_ceil_u32(self.context_dim, 256), 1, 1],
        )?;
        batch.readback_f32(&self.row_loss, &self.row_loss_readback, backward.rows)?;
        Ok(HierarchosOutNormRecordedStep {
            rows: backward.rows,
            lm_step,
            next_norm_step: next_step,
        })
    }

    /// Canonical trainable view for the one-registry full-model AdamW path.
    /// `lm_head.weight` points at the shared tied gradient accumulator, which
    /// already contains dense LM plus every recurrent DeepEmbed contribution.
    pub(crate) fn optimizer_trainables(&self) -> Vec<rwkv_optimizer::RwkvTrainableRef<'_>> {
        use rwkv_optimizer::{RwkvDecayClass, RwkvTrainableRef};

        vec![
            RwkvTrainableRef {
                name: "out_norm.weight",
                parameter: &self.norm_weight,
                gradient: &self.grad_norm_weight,
                len: self.context_dim,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "out_norm.bias",
                parameter: &self.norm_bias,
                gradient: &self.grad_norm_bias,
                len: self.context_dim,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "lm_head.weight",
                parameter: self.lm_head.weight_buffer(),
                gradient: self.lm_head.gradient_buffer(),
                len: self.context_dim * self.vocab_size,
                decay_class: RwkvDecayClass::Decay,
            },
        ]
    }

    /// Finish output bookkeeping after an external full-model optimizer has
    /// stepped every parameter. This deliberately skips the legacy out_norm and
    /// shared-LM optimizer islands while preserving loss readback/finalization.
    pub(crate) fn record_finish_with_external_optimizer(
        &self,
        batch: &mut vulkan::ComputeBatch,
        backward: HierarchosOutNormBackwardTicket,
        optimizer_step: u32,
    ) -> Result<HierarchosOutNormRecordedStep> {
        batch.readback_f32(&self.row_loss, &self.row_loss_readback, backward.rows)?;
        Ok(HierarchosOutNormRecordedStep {
            rows: backward.rows,
            lm_step: optimizer_step,
            next_norm_step: optimizer_step,
        })
    }

    /// Finish output bookkeeping for a caller-owned sequence command buffer,
    /// copying this token's row losses into a caller-owned host-visible slot.
    /// Unlike `record_finish_with_external_optimizer`, this does not touch the
    /// trainer-global loss readback, so a later token may safely reuse the
    /// output scratch buffers before the command buffer is submitted.
    pub(crate) fn record_finish_with_external_optimizer_into(
        &self,
        batch: &mut vulkan::ComputeBatch,
        backward: HierarchosOutNormBackwardTicket,
        row_loss_readback: &GpuBuffer,
    ) -> Result<()> {
        batch.readback_f32(&self.row_loss, row_loss_readback, backward.rows)
    }

    /// Commit host-side output-step bookkeeping after a caller-owned command
    /// buffer containing the canonical full-model optimizer step has completed.
    pub(crate) fn finalize_external_optimizer_step(&mut self, optimizer_step: u32) {
        self.step = optimizer_step;
    }

    pub(crate) fn grad_input_buffer(&self) -> &GpuBuffer {
        &self.grad_input_hidden
    }

    /// Train the output slice while accumulating the gradient arriving through
    /// the tied token-embedding use of `lm_head.weight`. `token_ids` identifies
    /// each embedding lookup and `embedding_grad` is its upstream `[tokens,
    /// context_dim]` gradient from the body backward pass.
    pub fn train_step_with_tied_embedding_grad(
        &mut self,
        pre_norm_hidden: &[f32],
        targets: &[u32],
        token_ids: &[u32],
        embedding_grad: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<TrainStepResult> {
        self.train_step_internal(
            pre_norm_hidden,
            targets,
            Some((token_ids, embedding_grad)),
            hyper,
            true,
        )
    }

    fn record_tied_embedding_radix_sort(
        &self,
        batch: &mut vulkan::ComputeBatch,
        token_count: usize,
    ) -> Result<&GpuBuffer> {
        let block_count = token_count.div_ceil(EMBEDDING_RADIX_BLOCK_SIZE);
        let pass_count = embedding_radix_pass_count(self.vocab_size);
        for pass in 0..pass_count {
            let source_positions = if pass == 0 || pass.is_multiple_of(2) {
                &self.tied_sorted_token_positions
            } else {
                &self.tied_radix_scratch_positions
            };
            let destination_positions = if pass.is_multiple_of(2) {
                &self.tied_radix_scratch_positions
            } else {
                &self.tied_sorted_token_positions
            };
            let radix_push = EmbeddingRadixPush {
                token_count: token_count as u32,
                shift: (pass * 4) as u32,
                block_count: block_count as u32,
                source_is_identity: u32::from(pass == 0),
            };
            self.embedding_radix_histogram.record_dispatch(
                batch,
                &[
                    &self.tied_token_ids,
                    source_positions,
                    &self.tied_radix_block_histograms,
                ],
                bytemuck::bytes_of(&radix_push),
                [block_count as u32, 1, 1],
            )?;
            self.embedding_radix_prefix.record_dispatch(
                batch,
                &[
                    &self.tied_radix_block_histograms,
                    &self.tied_radix_block_offsets,
                ],
                bytemuck::bytes_of(&radix_push),
                [1, 1, 1],
            )?;
            self.embedding_radix_scatter.record_dispatch(
                batch,
                &[
                    &self.tied_token_ids,
                    source_positions,
                    &self.tied_radix_block_offsets,
                    destination_positions,
                ],
                bytemuck::bytes_of(&radix_push),
                [block_count as u32, 1, 1],
            )?;
        }
        Ok(if pass_count.is_multiple_of(2) {
            &self.tied_sorted_token_positions
        } else {
            &self.tied_radix_scratch_positions
        })
    }

    fn train_step_internal(
        &mut self,
        pre_norm_hidden: &[f32],
        targets: &[u32],
        tied_embedding: Option<(&[u32], &[f32])>,
        hyper: AdamWHyperParams,
        reset_shared_lm_grad: bool,
    ) -> Result<TrainStepResult> {
        let mut batch = vulkan::ComputeBatch::new(&self.device)?;
        let recorded = self.record_train_step_internal(
            &mut batch,
            pre_norm_hidden,
            targets,
            tied_embedding,
            hyper,
            reset_shared_lm_grad,
        )?;
        batch.submit()?;
        self.finalize_recorded_train_step(recorded)
    }

    fn record_train_step_internal(
        &mut self,
        mut batch: &mut vulkan::ComputeBatch,
        pre_norm_hidden: &[f32],
        targets: &[u32],
        tied_embedding: Option<(&[u32], &[f32])>,
        hyper: AdamWHyperParams,
        reset_shared_lm_grad: bool,
    ) -> Result<HierarchosOutNormRecordedStep> {
        hyper.validate()?;
        if !pre_norm_hidden.len().is_multiple_of(self.context_dim) {
            bail!(
                "pre-norm hidden length {} is not divisible by context_dim {}",
                pre_norm_hidden.len(),
                self.context_dim
            );
        }
        let rows = pre_norm_hidden.len() / self.context_dim;
        if rows == 0 || rows > self.max_rows {
            bail!(
                "batch has {rows} rows; trainer capacity is 1..={}",
                self.max_rows
            );
        }
        if targets.len() != rows {
            bail!(
                "target count {} does not match batch rows {rows}",
                targets.len()
            );
        }
        if let Some(&bad) = targets
            .iter()
            .find(|&&target| target as usize >= self.vocab_size)
        {
            bail!(
                "target token {bad} is outside vocabulary size {}",
                self.vocab_size
            );
        }
        if pre_norm_hidden.iter().any(|value| !value.is_finite()) {
            bail!("pre-norm hidden batch contains non-finite values");
        }

        if let Some((token_ids, embedding_grad)) = tied_embedding {
            if token_ids.is_empty() || token_ids.len() > self.max_rows {
                bail!(
                    "tied embedding token count must be in 1..={}; got {}",
                    self.max_rows,
                    token_ids.len()
                );
            }
            let expected_grad = token_ids
                .len()
                .checked_mul(self.context_dim)
                .context("tied embedding gradient size overflow")?;
            if embedding_grad.len() != expected_grad {
                bail!(
                    "tied embedding gradient has {} values; expected {} for [{}, {}]",
                    embedding_grad.len(),
                    expected_grad,
                    token_ids.len(),
                    self.context_dim
                );
            }
            if let Some(&bad) = token_ids
                .iter()
                .find(|&&token| token as usize >= self.vocab_size)
            {
                bail!(
                    "tied embedding token {bad} is outside vocabulary size {}",
                    self.vocab_size
                );
            }
            if embedding_grad.iter().any(|value| !value.is_finite()) {
                bail!("tied embedding gradient contains non-finite values");
            }
            batch.upload_u32(&self.tied_token_ids, token_ids)?;
            batch.upload_f32(&self.tied_embedding_grad, embedding_grad)?;
        }

        batch.upload_f32(&self.input_hidden, pre_norm_hidden)?;
        batch.upload_u32(&self.targets, targets)?;
        if reset_shared_lm_grad {
            self.lm_head.record_zero_grad(&mut batch)?;
        }

        // Streaming CE forward deliberately preserves the fifth row-stat
        // scalar because it belongs to the caller: every backward consumer
        // multiplies its CE adjoint by this source-gradient coefficient. The
        // standalone trainer uses PyTorch's reduction="mean" contract, so seed
        // exactly 1 / rows here just like the split/full-model entrypoint does.
        // Leaving this buffer at its zero-initialized value silently erased the
        // dense LM and out_norm gradients while still producing correct losses.
        let mean_ce_source_weight = 1.0 / rows as f32;
        let mut ce_row_stats = vec![0.0f32; rows * 5];
        for row in 0..rows {
            ce_row_stats[row * 5 + 4] = mean_ce_source_weight;
        }
        batch.upload_f32(&self.ce_row_stats, &ce_row_stats)?;

        let linear_push = self.record_norm_head_stats(&mut batch, &self.input_hidden, rows)?;
        self.record_streaming_lm_loss_backward(&mut batch, &self.input_hidden, rows, linear_push)?;
        if let Some((token_ids, _)) = tied_embedding {
            let embedding_push = EmbeddingGradPush {
                token_count: token_ids.len() as u32,
                dim: self.context_dim as u32,
                vocab_size: self.vocab_size as u32,
            };
            let sorted_positions = if token_ids.len() <= EMBEDDING_SEGMENTED_SORT_CAPACITY {
                self.embedding_token_sort.record_dispatch(
                    &mut batch,
                    &[&self.tied_token_ids, &self.tied_sorted_token_positions],
                    bytemuck::bytes_of(&embedding_push),
                    [1, 1, 1],
                )?;
                &self.tied_sorted_token_positions
            } else {
                self.record_tied_embedding_radix_sort(&mut batch, token_ids.len())?
            };
            self.embedding_grad_segmented.record_dispatch(
                &mut batch,
                &[
                    &self.tied_token_ids,
                    sorted_positions,
                    &self.tied_embedding_grad,
                    self.lm_head.gradient_buffer(),
                ],
                bytemuck::bytes_of(&embedding_push),
                [
                    div_ceil_u32(self.context_dim, 16),
                    div_ceil_u32(token_ids.len(), 16),
                    1,
                ],
            )?;
        }

        self.record_out_norm_backward(&mut batch, &self.input_hidden, rows, false)?;

        let next_step = self.step.checked_add(1).context("AdamW step overflow")?;
        let lm_step = self.lm_head.record_step(&mut batch, hyper)?;
        let norm_adam_push = AdamWPush {
            len: self.context_dim as u32,
            step: next_step,
            lr: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay: 0.0,
        };
        self.adamw.record_dispatch(
            &mut batch,
            &[
                &self.norm_weight,
                &self.grad_norm_weight,
                &self.norm_weight_exp_avg,
                &self.norm_weight_exp_avg_sq,
            ],
            bytemuck::bytes_of(&norm_adam_push),
            [div_ceil_u32(self.context_dim, 256), 1, 1],
        )?;
        self.adamw.record_dispatch(
            &mut batch,
            &[
                &self.norm_bias,
                &self.grad_norm_bias,
                &self.norm_bias_exp_avg,
                &self.norm_bias_exp_avg_sq,
            ],
            bytemuck::bytes_of(&norm_adam_push),
            [div_ceil_u32(self.context_dim, 256), 1, 1],
        )?;
        batch.readback_f32(&self.row_loss, &self.row_loss_readback, rows)?;
        Ok(HierarchosOutNormRecordedStep {
            rows,
            lm_step,
            next_norm_step: next_step,
        })
    }

    pub fn input_grad(&self, rows: usize) -> Result<Vec<f32>> {
        if rows == 0 || rows > self.max_rows {
            bail!("input-gradient rows must be in 1..={}", self.max_rows);
        }
        self.grad_input_hidden.read_f32(rows * self.context_dim)
    }

    pub fn lm_weights(&self) -> Result<Vec<f32>> {
        self.lm_head.weights()
    }

    pub fn shared_lm_head(&self) -> SharedLmHeadParameter {
        self.lm_head.clone()
    }

    pub fn norm_weights(&self) -> Result<Vec<f32>> {
        self.norm_weight.read_f32(self.context_dim)
    }

    pub fn norm_bias(&self) -> Result<Vec<f32>> {
        self.norm_bias.read_f32(self.context_dim)
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn export_model_package(
        &self,
        source_model_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
    ) -> Result<()> {
        let source_model_dir = source_model_dir.as_ref();
        let output_dir = output_dir.as_ref();
        if source_model_dir == output_dir {
            bail!("export_model_package requires a distinct output directory");
        }
        let config = ModelConfig::from_model_dir(source_model_dir)
            .context("validating source model package")?;
        if config.context_dim != self.context_dim || config.vocab_size != self.vocab_size {
            bail!(
                "trainer shape [{}, {}] does not match model package [{}, {}]",
                self.vocab_size,
                self.context_dim,
                config.vocab_size,
                config.context_dim
            );
        }
        std::fs::create_dir_all(output_dir)?;
        for entry in std::fs::read_dir(source_model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path.file_name().and_then(|name| name.to_str()) != Some("model.safetensors")
            {
                std::fs::copy(&path, output_dir.join(entry.file_name()))?;
            }
        }
        let lm_weight = self.lm_weights()?;
        let norm_weight = self.norm_weights()?;
        let norm_bias = self.norm_bias()?;
        replace_f32_tensors(
            &source_model_dir.join("model.safetensors"),
            &output_dir.join("model.safetensors"),
            &[
                (
                    "lm_head.weight",
                    &[self.vocab_size, self.context_dim],
                    &lm_weight,
                ),
                ("out_norm.weight", &[self.context_dim], &norm_weight),
                ("out_norm.bias", &[self.context_dim], &norm_bias),
            ],
        )?;
        Ok(())
    }
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

fn embedding_radix_pass_count(vocab_size: usize) -> usize {
    let highest_token = u32::try_from(vocab_size.saturating_sub(1)).unwrap_or(u32::MAX);
    let significant_bits = (u32::BITS - highest_token.leading_zeros()).max(1) as usize;
    significant_bits.div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HIERARCHOS_VULKAN_MICROPROFILE_DEVICE_INDEX_ENV: &str =
        "HIERARCHOS_VULKAN_MICROPROFILE_DEVICE_INDEX";

    fn microprofile_device() -> Result<Option<VulkanDevice>> {
        match std::env::var(HIERARCHOS_VULKAN_MICROPROFILE_DEVICE_INDEX_ENV) {
            Ok(raw) => {
                let index = raw.parse::<usize>().with_context(|| {
                    format!(
                        "{HIERARCHOS_VULKAN_MICROPROFILE_DEVICE_INDEX_ENV} must be a non-negative physical-device index, got {raw:?}"
                    )
                })?;
                Ok(Some(VulkanDevice::new_with_index(index).with_context(
                    || format!("initializing Vulkan microprofile physical-device index {index}"),
                )?))
            }
            Err(std::env::VarError::NotPresent) => Ok(VulkanDevice::new().ok()),
            Err(err) => Err(err).with_context(|| {
                format!("reading {HIERARCHOS_VULKAN_MICROPROFILE_DEVICE_INDEX_ENV}")
            }),
        }
    }

    fn mean_ce_row_stats_buffer(device: &VulkanDevice, rows: usize) -> Result<GpuBuffer> {
        let mut stats = vec![0.0f32; rows * 5];
        let weight = 1.0 / rows as f32;
        for row in 0..rows {
            stats[row * 5 + 4] = weight;
        }
        GpuBuffer::from_f32(device, &stats)
    }

    #[test]
    fn weighted_ce_adjoint_respects_zero_and_fractional_row_mass() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let rows = 3usize;
        let vocab = 4usize;
        let logits = vec![
            1.25, -0.5, 0.75, 0.0, -0.25, 0.5, 1.5, -1.0, 0.2, -0.1, 0.4, 0.8,
        ];
        let targets = [0u32, 2u32, 3u32];
        let row_weights = [1.0f32, 0.0, 0.25];
        let mut stats = vec![0.0f32; rows * 5];
        let mut expected = vec![0.0f32; rows * vocab];
        for row in 0..rows {
            let row_logits = &logits[row * vocab..(row + 1) * vocab];
            let max_logit = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum = row_logits
                .iter()
                .map(|&value| (value - max_logit).exp())
                .sum::<f32>();
            stats[row * 5] = max_logit;
            stats[row * 5 + 1] = 1.0 / exp_sum;
            stats[row * 5 + 2] = f32::from_bits(targets[row]);
            stats[row * 5 + 4] = row_weights[row];
            for col in 0..vocab {
                let probability = (row_logits[col] - max_logit).exp() / exp_sum;
                expected[row * vocab + col] = (probability
                    - if col == targets[row] as usize {
                        1.0
                    } else {
                        0.0
                    })
                    * row_weights[row];
            }
        }

        let row_stats = GpuBuffer::from_f32(&device, &stats)?;
        let logit_tape = GpuBuffer::from_f32(&device, &logits)?;
        let kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LOGITS_TO_GRAD_INPLACE_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: 1,
            output_dim: vocab as u32,
            z_loss_weight: 0.0,
            activation_clamp: f32::MAX,
        };
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&row_stats, &logit_tape],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(rows * vocab, 128), 1, 1],
        )?;
        commands.submit()?;

        let actual = logit_tape.read_f32(rows * vocab)?;
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 2.0e-6,
            "weighted CE adjoint drifted by {max_abs}"
        );
        assert!(actual[vocab..2 * vocab].iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn gpu_buffer_clone_keeps_one_allocation_alive() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let buffer = GpuBuffer::from_f32(&device, &[1.0, 2.0, 3.0, 4.0])?;
        let alias = buffer.clone();
        assert!(buffer.shares_allocation_with(&alias));
        drop(buffer);
        assert_eq!(alias.read_f32(4)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn persistent_upload_arena_reuses_staging_allocation_across_batches() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let arena = vulkan::PersistentUploadArena::new();
        let dst = GpuBuffer::zeros_f32(&device, 4)?;

        for values in [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] {
            let mut commands = vulkan::ComputeBatch::new_with_persistent_upload_arena(
                &device,
                std::sync::Arc::clone(&arena),
            )?;
            commands.upload_f32(&dst, &values)?;
            assert_eq!(commands.upload_arena_buffer_count(), 1);
            commands.submit()?;
            assert_eq!(arena.buffer_count()?, 1);
        }

        Ok(())
    }

    #[test]
    fn compatible_compute_kernels_share_interned_layout_handles() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let forward = vulkan::ComputeKernel::new(
            &device,
            LINEAR_FORWARD_SPV,
            3,
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let input_grad = vulkan::ComputeKernel::new_with_access(
            &device,
            LINEAR_INPUT_GRAD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;

        assert!(forward.shares_interned_layouts_with(&input_grad));
        Ok(())
    }

    #[test]
    fn fused_out_norm_head_forward_matches_legacy_and_removes_dispatch_seam() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        if !device.supports_storage_buffer_bindings(8) {
            return Ok(());
        }

        let rows = 3usize;
        let input_dim = 5usize;
        let output_dim = 7usize;
        let input = GpuBuffer::from_f32(
            &device,
            &[
                0.25, -0.75, 0.50, 1.25, -0.10, -0.40, 0.80, 0.15, -1.10, 0.65, 1.50, -0.20, 0.35,
                -0.95, 0.45,
            ],
        )?;
        let norm_weight = GpuBuffer::from_f32(&device, &[1.0, 0.8, 1.2, 0.9, 1.1])?;
        let norm_bias = GpuBuffer::from_f32(&device, &[0.05, -0.03, 0.02, 0.01, -0.04])?;
        let linear_weight = GpuBuffer::from_f32(
            &device,
            &[
                0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80, 0.90, -1.00, 0.15, 0.25, -0.35,
                0.45, -0.55, 0.65, -0.75, 0.85, -0.95, 1.05, 0.12, -0.22, 0.32, -0.42, 0.52, -0.62,
                0.72, -0.82, 0.92, -1.02, 0.18, 0.28, -0.38, 0.48, -0.58,
            ],
        )?;

        let legacy_norm = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
        let legacy_mean = GpuBuffer::zeros_f32(&device, rows)?;
        let legacy_rstd = GpuBuffer::zeros_f32(&device, rows)?;
        let legacy_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let fused_norm = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
        let fused_mean = GpuBuffer::zeros_f32(&device, rows)?;
        let fused_rstd = GpuBuffer::zeros_f32(&device, rows)?;
        let fused_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;

        let layer_norm = vulkan::ComputeKernel::new(
            &device,
            LAYER_NORM_FORWARD_SPV,
            6,
            std::mem::size_of::<LayerNormForwardPush>() as u32,
        )?;
        let linear = vulkan::ComputeKernel::new(
            &device,
            LINEAR_FORWARD_SPV,
            3,
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let fused = vulkan::ComputeKernel::new_with_access(
            &device,
            LAYER_NORM_LINEAR_FORWARD_FUSED_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LayerNormLinearForwardPush>() as u32,
        )?;

        let norm_push = LayerNormForwardPush {
            rows: rows as u32,
            dim: input_dim as u32,
            eps: HierarchosOutNormHeadTrainer::LAYER_NORM_EPS,
        };
        let linear_push = LinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            z_loss_weight: 0.0,
            activation_clamp: f32::MAX,
        };
        let mut legacy_batch = vulkan::ComputeBatch::new(&device)?;
        layer_norm.record_dispatch(
            &mut legacy_batch,
            &[
                &input,
                &norm_weight,
                &norm_bias,
                &legacy_norm,
                &legacy_mean,
                &legacy_rstd,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        linear.record_dispatch(
            &mut legacy_batch,
            &[&legacy_norm, &linear_weight, &legacy_logits],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;
        assert_eq!(legacy_batch.dispatch_count(), 2);
        assert_eq!(legacy_batch.shader_barrier_count(), 1);
        legacy_batch.submit()?;

        let fused_push = LayerNormLinearForwardPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            eps: HierarchosOutNormHeadTrainer::LAYER_NORM_EPS,
        };
        let mut fused_batch = vulkan::ComputeBatch::new(&device)?;
        fused.record_dispatch(
            &mut fused_batch,
            &[
                &input,
                &norm_weight,
                &norm_bias,
                &linear_weight,
                &fused_norm,
                &fused_mean,
                &fused_rstd,
                &fused_logits,
            ],
            bytemuck::bytes_of(&fused_push),
            [div_ceil_u32(output_dim, 64), rows as u32, 1],
        )?;
        assert_eq!(fused_batch.dispatch_count(), 1);
        assert_eq!(fused_batch.shader_barrier_count(), 0);
        fused_batch.submit()?;

        for (name, legacy_values, fused_values) in [
            (
                "normalized hidden",
                legacy_norm.read_f32(rows * input_dim)?,
                fused_norm.read_f32(rows * input_dim)?,
            ),
            (
                "mean",
                legacy_mean.read_f32(rows)?,
                fused_mean.read_f32(rows)?,
            ),
            (
                "rstd",
                legacy_rstd.read_f32(rows)?,
                fused_rstd.read_f32(rows)?,
            ),
            (
                "logits",
                legacy_logits.read_f32(rows * output_dim)?,
                fused_logits.read_f32(rows * output_dim)?,
            ),
        ] {
            let max_abs = legacy_values
                .iter()
                .zip(&fused_values)
                .map(|(legacy, fused)| (legacy - fused).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= 1.0e-6, "fused {name} drifted by {max_abs}");
        }

        Ok(())
    }

    #[test]
    fn fused_cross_entropy_lm_backward_matches_materialized_gradients() -> Result<()> {
        const LEGACY_CROSS_ENTROPY_SPV: &[u8] = include_bytes!("../shaders/cross_entropy_grad.spv");

        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let rows = 3usize;
        let input_dim = 5usize;
        let output_dim = 7usize;
        let hidden_values = [
            0.25, -0.75, 0.50, 1.25, -0.10, -0.40, 0.80, 0.15, -1.10, 0.65, 1.50, -0.20, 0.35,
            -0.95, 0.45,
        ];
        let logits_values = [
            1.25, -0.75, 0.50, 2.00, -1.50, 0.10, 0.90, -0.25, 1.10, 0.35, -0.80, 0.65, 1.75,
            -1.20, 0.40, -0.60, 1.30, 0.20, -0.10, 0.85, -1.40,
        ];
        let weight_values = [
            0.10, -0.20, 0.30, 0.40, -0.50, -0.60, 0.70, 0.80, -0.90, 1.00, 1.10, -1.20, 1.30,
            1.40, -1.50, -1.60, 1.70, 1.80, -1.90, 2.00, 2.10, -2.20, 2.30, 2.40, -2.50, -2.60,
            2.70, 2.80, -2.90, 3.00, 3.10, -3.20, 3.30, 3.40, -3.50,
        ];
        let targets = [3u32, 5u32, 1u32];

        let hidden = GpuBuffer::from_f32(&device, &hidden_values)?;
        let logits = GpuBuffer::from_f32(&device, &logits_values)?;
        let weights = GpuBuffer::from_f32(&device, &weight_values)?;
        let targets_buffer = GpuBuffer::from_u32(&device, &targets)?;

        let legacy_grad_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let legacy_loss = GpuBuffer::zeros_f32(&device, rows)?;
        let legacy_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let legacy_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
        let fused_stats = GpuBuffer::zeros_f32(&device, rows * 3)?;
        let fused_loss = GpuBuffer::zeros_f32(&device, rows)?;
        let fused_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let fused_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;

        let legacy_cross_entropy = vulkan::ComputeKernel::new(
            &device,
            LEGACY_CROSS_ENTROPY_SPV,
            4,
            std::mem::size_of::<CrossEntropyPush>() as u32,
        )?;
        let fused_cross_entropy = vulkan::ComputeKernel::new(
            &device,
            CROSS_ENTROPY_ROW_STATS_SPV,
            4,
            std::mem::size_of::<CrossEntropyPush>() as u32,
        )?;
        let legacy_weight_grad = vulkan::ComputeKernel::new_with_access(
            &device,
            LINEAR_WEIGHT_GRAD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<WeightGradPush>() as u32,
        )?;
        let fused_weight_grad = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<WeightGradPush>() as u32,
        )?;
        let legacy_input_grad = vulkan::ComputeKernel::new_with_access(
            &device,
            LINEAR_INPUT_GRAD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let fused_input_grad = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_INPUT_GRAD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;

        let xent_push = CrossEntropyPush {
            rows: rows as u32,
            vocab_size: output_dim as u32,
            z_loss_weight: 0.0,
        };
        let linear_push = LinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            z_loss_weight: 0.0,
            activation_clamp: f32::MAX,
        };
        let weight_push = WeightGradPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        };

        let mut legacy_batch = vulkan::ComputeBatch::new(&device)?;
        legacy_cross_entropy.record_dispatch(
            &mut legacy_batch,
            &[&logits, &targets_buffer, &legacy_grad_logits, &legacy_loss],
            bytemuck::bytes_of(&xent_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        legacy_weight_grad.record_dispatch(
            &mut legacy_batch,
            &[&hidden, &legacy_grad_logits, &legacy_grad_weight],
            bytemuck::bytes_of(&weight_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(output_dim, 16), 1],
        )?;
        legacy_input_grad.record_dispatch(
            &mut legacy_batch,
            &[&legacy_grad_logits, &weights, &legacy_grad_input],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;
        legacy_batch.submit()?;

        let mut fused_batch = vulkan::ComputeBatch::new(&device)?;
        fused_cross_entropy.record_dispatch(
            &mut fused_batch,
            &[&logits, &targets_buffer, &fused_stats, &fused_loss],
            bytemuck::bytes_of(&xent_push),
            [rows as u32, 1, 1],
        )?;
        fused_weight_grad.record_dispatch(
            &mut fused_batch,
            &[&hidden, &logits, &fused_stats, &fused_grad_weight],
            bytemuck::bytes_of(&weight_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(output_dim, 16), 1],
        )?;
        fused_input_grad.record_dispatch(
            &mut fused_batch,
            &[&logits, &fused_stats, &weights, &fused_grad_input],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;
        fused_batch.submit()?;

        for (name, legacy_values, fused_values, tolerance) in [
            (
                "row loss",
                legacy_loss.read_f32(rows)?,
                fused_loss.read_f32(rows)?,
                1.0e-6f32,
            ),
            (
                "LM weight gradient",
                legacy_grad_weight.read_f32(output_dim * input_dim)?,
                fused_grad_weight.read_f32(output_dim * input_dim)?,
                2.0e-6f32,
            ),
            (
                "LM input gradient",
                legacy_grad_input.read_f32(rows * input_dim)?,
                fused_grad_input.read_f32(rows * input_dim)?,
                2.0e-6f32,
            ),
        ] {
            let max_abs = legacy_values
                .iter()
                .zip(&fused_values)
                .map(|(legacy, fused)| (legacy - fused).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= tolerance, "fused {name} drifted by {max_abs}");
        }

        Ok(())
    }

    #[test]
    fn vocabulary_streaming_lm_loss_matches_materialized_logits_across_tile_tail() -> Result<()> {
        const LEGACY_CROSS_ENTROPY_SPV: &[u8] = include_bytes!("../shaders/cross_entropy_grad.spv");

        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let rows = 3usize;
        let input_dim = 5usize;
        let output_dim = 73usize;
        let hidden_values = (0..rows * input_dim)
            .map(|index| ((index as f32 * 0.37).sin() * 0.8) + (index as f32 * 0.01))
            .collect::<Vec<_>>();
        let weight_values = (0..output_dim * input_dim)
            .map(|index| ((index as f32 * 0.19).cos() * 0.35) - (index % 7) as f32 * 0.005)
            .collect::<Vec<_>>();
        let norm_weight_values = (0..input_dim)
            .map(|index| 0.85 + index as f32 * 0.07)
            .collect::<Vec<_>>();
        let norm_bias_values = (0..input_dim)
            .map(|index| (index as f32 * 0.23).sin() * 0.1)
            .collect::<Vec<_>>();
        let targets = [0u32, 37u32, 72u32];

        let hidden = GpuBuffer::from_f32(&device, &hidden_values)?;
        let weights = GpuBuffer::from_f32(&device, &weight_values)?;
        let norm_weight = GpuBuffer::from_f32(&device, &norm_weight_values)?;
        let norm_bias = GpuBuffer::from_f32(&device, &norm_bias_values)?;
        let targets_buffer = GpuBuffer::from_u32(&device, &targets)?;

        let legacy_norm_hidden = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
        let legacy_norm_mean = GpuBuffer::zeros_f32(&device, rows)?;
        let legacy_norm_rstd = GpuBuffer::zeros_f32(&device, rows)?;
        let legacy_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let legacy_grad_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let legacy_loss = GpuBuffer::zeros_f32(&device, rows)?;
        let legacy_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let legacy_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;

        let streaming_norm_mean = GpuBuffer::zeros_f32(&device, rows)?;
        let streaming_norm_rstd = GpuBuffer::zeros_f32(&device, rows)?;
        let streaming_stats = mean_ce_row_stats_buffer(&device, rows)?;
        let streaming_loss = GpuBuffer::zeros_f32(&device, rows)?;
        let streaming_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let streaming_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;

        let layer_norm_forward = vulkan::ComputeKernel::new(
            &device,
            LAYER_NORM_FORWARD_SPV,
            6,
            std::mem::size_of::<LayerNormForwardPush>() as u32,
        )?;
        let layer_norm_stats = vulkan::ComputeKernel::new_with_access(
            &device,
            LAYER_NORM_STATS_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LayerNormForwardPush>() as u32,
        )?;
        let linear = vulkan::ComputeKernel::new(
            &device,
            LINEAR_FORWARD_SPV,
            3,
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let legacy_cross_entropy = vulkan::ComputeKernel::new(
            &device,
            LEGACY_CROSS_ENTROPY_SPV,
            4,
            std::mem::size_of::<CrossEntropyPush>() as u32,
        )?;
        let legacy_weight_grad = vulkan::ComputeKernel::new_with_access(
            &device,
            LINEAR_WEIGHT_GRAD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<WeightGradPush>() as u32,
        )?;
        let legacy_input_grad = vulkan::ComputeKernel::new_with_access(
            &device,
            LINEAR_INPUT_GRAD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let streaming_stats_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let streaming_loss_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_ROW_LOSS_EXTRACT_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<CrossEntropyPush>() as u32,
        )?;
        let streaming_weight_grad_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadWrite,
            ],
            std::mem::size_of::<LmWeightGradPush>() as u32,
        )?;
        let streaming_input_grad_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;

        let xent_push = CrossEntropyPush {
            rows: rows as u32,
            vocab_size: output_dim as u32,
            z_loss_weight: 0.0,
        };
        let linear_push = LinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            z_loss_weight: 0.0,
            activation_clamp: f32::MAX,
        };
        let lm_weight_grad_push = LmWeightGradPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            accumulate: 0,
            z_loss_weight: 0.0,
            activation_clamp: linear_push.activation_clamp,
        };
        let weight_push = WeightGradPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        };
        let norm_push = LayerNormForwardPush {
            rows: rows as u32,
            dim: input_dim as u32,
            eps: HierarchosOutNormHeadTrainer::LAYER_NORM_EPS,
        };

        let mut legacy_batch = vulkan::ComputeBatch::new(&device)?;
        layer_norm_forward.record_dispatch(
            &mut legacy_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &legacy_norm_hidden,
                &legacy_norm_mean,
                &legacy_norm_rstd,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        linear.record_dispatch(
            &mut legacy_batch,
            &[&legacy_norm_hidden, &weights, &legacy_logits],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;
        legacy_cross_entropy.record_dispatch(
            &mut legacy_batch,
            &[
                &legacy_logits,
                &targets_buffer,
                &legacy_grad_logits,
                &legacy_loss,
            ],
            bytemuck::bytes_of(&xent_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        legacy_weight_grad.record_dispatch(
            &mut legacy_batch,
            &[
                &legacy_norm_hidden,
                &legacy_grad_logits,
                &legacy_grad_weight,
            ],
            bytemuck::bytes_of(&weight_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(output_dim, 16), 1],
        )?;
        legacy_input_grad.record_dispatch(
            &mut legacy_batch,
            &[&legacy_grad_logits, &weights, &legacy_grad_input],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;
        legacy_batch.submit()?;

        let mut streaming_batch = vulkan::ComputeBatch::new(&device)?;
        layer_norm_stats.record_dispatch(
            &mut streaming_batch,
            &[&hidden, &streaming_norm_mean, &streaming_norm_rstd],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        streaming_stats_kernel.record_dispatch(
            &mut streaming_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &streaming_norm_mean,
                &streaming_norm_rstd,
                &weights,
                &targets_buffer,
                &streaming_stats,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        streaming_loss_kernel.record_dispatch(
            &mut streaming_batch,
            &[&streaming_stats, &streaming_loss],
            bytemuck::bytes_of(&xent_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        streaming_weight_grad_kernel.record_dispatch(
            &mut streaming_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &streaming_norm_mean,
                &streaming_norm_rstd,
                &weights,
                &streaming_stats,
                &streaming_grad_weight,
            ],
            bytemuck::bytes_of(&lm_weight_grad_push),
            [
                div_ceil_u32(output_dim, CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_VOCAB_TILE),
                1,
                1,
            ],
        )?;
        streaming_input_grad_kernel.record_dispatch(
            &mut streaming_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &streaming_norm_mean,
                &streaming_norm_rstd,
                &weights,
                &streaming_stats,
                &streaming_grad_input,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        streaming_batch.submit()?;

        for (name, legacy_values, streaming_values, tolerance) in [
            (
                "row loss",
                legacy_loss.read_f32(rows)?,
                streaming_loss.read_f32(rows)?,
                2.0e-6f32,
            ),
            (
                "LM weight gradient",
                legacy_grad_weight.read_f32(output_dim * input_dim)?,
                streaming_grad_weight.read_f32(output_dim * input_dim)?,
                3.0e-6f32,
            ),
            (
                "LM input gradient",
                legacy_grad_input.read_f32(rows * input_dim)?,
                streaming_grad_input.read_f32(rows * input_dim)?,
                3.0e-6f32,
            ),
        ] {
            let max_abs = legacy_values
                .iter()
                .zip(&streaming_values)
                .map(|(legacy, streaming)| (legacy - streaming).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= tolerance,
                "vocabulary-streaming {name} drifted by {max_abs}"
            );
        }

        // Exercise the native packed-FP16 consumer as well. The reference
        // materializes the exact same quantized execution weights back to FP32,
        // so any disagreement here is kernel geometry/indexing drift rather
        // than ordinary FP16 quantization error against the master tensor.
        let fp16_weights = VulkanFp32MasterParameterMirror::new(
            device.clone(),
            VulkanParameterStorageFormat::Fp16,
            weight_values.len(),
        )?;
        fp16_weights.refresh_and_expand_from_fp32_master(&weights)?;

        let fp16_legacy_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let fp16_legacy_grad_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let fp16_legacy_loss = GpuBuffer::zeros_f32(&device, rows)?;
        let fp16_legacy_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let fp16_legacy_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
        let fp16_streaming_stats = mean_ce_row_stats_buffer(&device, rows)?;
        let fp16_streaming_grad_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let fp16_streaming_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let fp16_streaming_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;

        let fp16_streaming_stats_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let fp16_streaming_input_grad_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_PACKED_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let fp16_streaming_weight_grad_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadWrite,
            ],
            std::mem::size_of::<LmWeightGradPush>() as u32,
        )?;

        let mut fp16_legacy_batch = vulkan::ComputeBatch::new(&device)?;
        linear.record_dispatch(
            &mut fp16_legacy_batch,
            &[
                &legacy_norm_hidden,
                fp16_weights.expanded_f32(),
                &fp16_legacy_logits,
            ],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;
        legacy_cross_entropy.record_dispatch(
            &mut fp16_legacy_batch,
            &[
                &fp16_legacy_logits,
                &targets_buffer,
                &fp16_legacy_grad_logits,
                &fp16_legacy_loss,
            ],
            bytemuck::bytes_of(&xent_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        legacy_weight_grad.record_dispatch(
            &mut fp16_legacy_batch,
            &[
                &legacy_norm_hidden,
                &fp16_legacy_grad_logits,
                &fp16_legacy_grad_weight,
            ],
            bytemuck::bytes_of(&weight_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(output_dim, 16), 1],
        )?;
        legacy_input_grad.record_dispatch(
            &mut fp16_legacy_batch,
            &[
                &fp16_legacy_grad_logits,
                fp16_weights.expanded_f32(),
                &fp16_legacy_grad_input,
            ],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(input_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;
        fp16_legacy_batch.submit()?;

        let mut fp16_streaming_batch = vulkan::ComputeBatch::new(&device)?;
        fp16_streaming_stats_kernel.record_dispatch(
            &mut fp16_streaming_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &streaming_norm_mean,
                &streaming_norm_rstd,
                fp16_weights.packed_storage(),
                &targets_buffer,
                &fp16_streaming_stats,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        fp16_streaming_input_grad_kernel.record_dispatch(
            &mut fp16_streaming_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &streaming_norm_mean,
                &streaming_norm_rstd,
                fp16_weights.packed_storage(),
                &fp16_streaming_stats,
                &fp16_streaming_grad_input,
                &fp16_streaming_grad_logits,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        fp16_streaming_weight_grad_kernel.record_dispatch(
            &mut fp16_streaming_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &streaming_norm_mean,
                &streaming_norm_rstd,
                &fp16_streaming_grad_logits,
                &fp16_streaming_grad_weight,
            ],
            bytemuck::bytes_of(&lm_weight_grad_push),
            [
                div_ceil_u32(
                    output_dim,
                    HierarchosLmWeightGradTopology::VocabRows8.vocab_rows_per_group() as usize,
                ),
                1,
                1,
            ],
        )?;
        fp16_streaming_batch.submit()?;

        if device
            .mixed_precision_capabilities()
            .native_fp16_storage_compute_ready()
        {
            let native_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
            let native_grad_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
            let native_input_grad_kernel = vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?;
            let mut native_batch = vulkan::ComputeBatch::new(&device)?;
            native_input_grad_kernel.record_dispatch(
                &mut native_batch,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &streaming_norm_mean,
                    &streaming_norm_rstd,
                    fp16_weights.packed_storage(),
                    &fp16_streaming_stats,
                    &native_grad_input,
                    &native_grad_logits,
                ],
                bytemuck::bytes_of(&linear_push),
                [rows as u32, 1, 1],
            )?;
            native_batch.submit()?;

            for (name, packed, native) in [
                (
                    "input gradient",
                    fp16_streaming_grad_input.read_f32(rows * input_dim)?,
                    native_grad_input.read_f32(rows * input_dim)?,
                ),
                (
                    "CE adjoint tape",
                    fp16_streaming_grad_logits.read_f32(rows * output_dim)?,
                    native_grad_logits.read_f32(rows * output_dim)?,
                ),
            ] {
                let max_abs = packed
                    .iter()
                    .zip(&native)
                    .map(|(packed, native)| (packed - native).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 1.0e-7,
                    "native-FP16 {name} drifted from packed-FP16 by {max_abs}"
                );
            }
        }

        let fp16_legacy_grad_weight = fp16_legacy_grad_weight.read_f32(output_dim * input_dim)?;
        let fp16_streaming_grad_weight =
            fp16_streaming_grad_weight.read_f32(output_dim * input_dim)?;
        let fp16_legacy_grad_input = fp16_legacy_grad_input.read_f32(rows * input_dim)?;
        let fp16_streaming_grad_input = fp16_streaming_grad_input.read_f32(rows * input_dim)?;
        let fp16_max_abs = fp16_legacy_grad_weight
            .iter()
            .zip(&fp16_streaming_grad_weight)
            .map(|(legacy, streaming)| (legacy - streaming).abs())
            .fold(0.0f32, f32::max);
        assert!(
            fp16_max_abs <= 4.0e-6,
            "packed-FP16 cooperative LM weight gradient drifted by {fp16_max_abs}"
        );
        let fp16_input_max_abs = fp16_legacy_grad_input
            .iter()
            .zip(&fp16_streaming_grad_input)
            .map(|(legacy, streaming)| (legacy - streaming).abs())
            .fold(0.0f32, f32::max);
        assert!(
            fp16_input_max_abs <= 4.0e-6,
            "packed-FP16 cooperative LM input gradient drifted by {fp16_input_max_abs}"
        );

        Ok(())
    }

    #[test]
    fn native_fp16_lm_input_grad_reuse_arms_match_packed_at_width_448() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        if !device
            .mixed_precision_capabilities()
            .native_fp16_storage_compute_ready()
        {
            return Ok(());
        }

        let rows = 2usize;
        let input_dim = 448usize;
        let output_dim = 73usize;
        let hidden_values = (0..rows * input_dim)
            .map(|index| ((index as f32 * 0.017).sin() * 0.55) + (index % 19) as f32 * 0.002)
            .collect::<Vec<_>>();
        let weight_values = (0..output_dim * input_dim)
            .map(|index| ((index as f32 * 0.013).cos() * 0.11) - (index % 23) as f32 * 0.0007)
            .collect::<Vec<_>>();
        let norm_weight_values = (0..input_dim)
            .map(|index| 0.92 + (index % 31) as f32 * 0.004)
            .collect::<Vec<_>>();
        let norm_bias_values = (0..input_dim)
            .map(|index| (index as f32 * 0.071).sin() * 0.025)
            .collect::<Vec<_>>();
        let targets = [0u32, 72u32];

        let hidden = GpuBuffer::from_f32(&device, &hidden_values)?;
        let weights = GpuBuffer::from_f32(&device, &weight_values)?;
        let norm_weight = GpuBuffer::from_f32(&device, &norm_weight_values)?;
        let norm_bias = GpuBuffer::from_f32(&device, &norm_bias_values)?;
        let targets_buffer = GpuBuffer::from_u32(&device, &targets)?;
        let norm_mean = GpuBuffer::zeros_f32(&device, rows)?;
        let norm_rstd = GpuBuffer::zeros_f32(&device, rows)?;
        let stats = mean_ce_row_stats_buffer(&device, rows)?;
        let packed_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
        let packed_grad_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;

        let fp16_weights = VulkanFp32MasterParameterMirror::new(
            device.clone(),
            VulkanParameterStorageFormat::Fp16,
            weight_values.len(),
        )?;
        fp16_weights.refresh_and_expand_from_fp32_master(&weights)?;

        let norm_stats_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            LAYER_NORM_STATS_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LayerNormForwardPush>() as u32,
        )?;
        let row_stats_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let packed_input_grad_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_PACKED_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let norm_push = LayerNormForwardPush {
            rows: rows as u32,
            dim: input_dim as u32,
            eps: HierarchosOutNormHeadTrainer::LAYER_NORM_EPS,
        };
        let linear_push = LinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            z_loss_weight: 0.0,
            activation_clamp: f32::MAX,
        };
        let lm_weight_grad_push = LmWeightGradPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            accumulate: 0,
            z_loss_weight: 0.0,
            activation_clamp: linear_push.activation_clamp,
        };

        let mut packed_batch = vulkan::ComputeBatch::new(&device)?;
        norm_stats_kernel.record_dispatch(
            &mut packed_batch,
            &[&hidden, &norm_mean, &norm_rstd],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;
        row_stats_kernel.record_dispatch(
            &mut packed_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &norm_mean,
                &norm_rstd,
                fp16_weights.packed_storage(),
                &targets_buffer,
                &stats,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        packed_input_grad_kernel.record_dispatch(
            &mut packed_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &norm_mean,
                &norm_rstd,
                fp16_weights.packed_storage(),
                &stats,
                &packed_grad_input,
                &packed_grad_logits,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        packed_batch.submit()?;
        let packed_stats = stats.read_f32(rows * 5)?;
        let packed_input = packed_grad_input.read_f32(rows * input_dim)?;
        let packed_logits = packed_grad_logits.read_f32(rows * output_dim)?;

        let tape_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
        let tape_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let tape_logits_grad = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let tape_row_stats_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_ROW_STATS_STREAMING_FP16_PACKED_TAPE_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let tape_to_grad_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LOGITS_TO_GRAD_INPLACE_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let tape_input_grad_kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_INPUT_GRAD_TAPE_FP16_PACKED_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let tape_weight_grad_kernel = create_fp16_lm_tape_weight_grad_kernel(
            &device,
            HierarchosLmWeightGradTopology::VocabRows8,
            CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_SPV,
        )?
        .context("width-448 direct-logit dW vocab8 topology is unavailable")?;
        let mut tape_batch = vulkan::ComputeBatch::new(&device)?;
        tape_row_stats_kernel.record_dispatch(
            &mut tape_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &norm_mean,
                &norm_rstd,
                fp16_weights.packed_storage(),
                &targets_buffer,
                &stats,
                &tape_logits_grad,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        tape_input_grad_kernel.record_dispatch(
            &mut tape_batch,
            &[
                &tape_logits_grad,
                &stats,
                fp16_weights.packed_storage(),
                &tape_grad_input,
            ],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(input_dim.div_ceil(2), 32), 1, 1],
        )?;
        tape_weight_grad_kernel.record_dispatch(
            &mut tape_batch,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &norm_mean,
                &norm_rstd,
                &tape_logits_grad,
                &stats,
                &tape_grad_weight,
            ],
            bytemuck::bytes_of(&lm_weight_grad_push),
            [
                div_ceil_u32(
                    output_dim,
                    HierarchosLmWeightGradTopology::VocabRows8.vocab_rows_per_group() as usize,
                ),
                1,
                1,
            ],
        )?;
        // Retain the old rewrite only as a regression oracle. Production tape
        // backward has already consumed the untouched logits above.
        tape_to_grad_kernel.record_dispatch(
            &mut tape_batch,
            &[&stats, &tape_logits_grad],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(rows * output_dim, 128), 1, 1],
        )?;
        tape_batch.submit()?;
        let tape_input = tape_grad_input.read_f32(rows * input_dim)?;
        let tape_logits = tape_logits_grad.read_f32(rows * output_dim)?;
        let tape_input_max_abs = packed_input
            .iter()
            .zip(&tape_input)
            .map(|(packed, tape)| (packed - tape).abs())
            .fold(0.0f32, f32::max);
        let tape_logits_max_abs = packed_logits
            .iter()
            .zip(&tape_logits)
            .map(|(packed, tape)| (packed - tape).abs())
            .fold(0.0f32, f32::max);
        assert!(
            tape_input_max_abs <= 1.0e-7,
            "fp16-ce-tape width-448 input gradient drifted from packed FP16 by {tape_input_max_abs}"
        );
        assert!(
            tape_logits_max_abs <= 1.0e-7,
            "fp16-ce-tape width-448 CE adjoint drifted from packed FP16 by {tape_logits_max_abs}"
        );

        if input_dim <= 448
            && device.max_compute_shared_memory_bytes() >= 16_160
            && device.supports_compute_work_group_size_x(64)
            && device.supports_storage_buffer_bindings(9)
        {
            let rows8_stats = mean_ce_row_stats_buffer(&device, rows)?;
            let rows8_logits_grad = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
            let rows8_tile_stats =
                GpuBuffer::zeros_f32(&device, rows * output_dim.div_ceil(8) * 3)?;
            let rows8_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
            let rows8_projection_kernel = vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS8_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?;
            let rows8_stats_kernel = vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?;
            let mut rows8_batch = vulkan::ComputeBatch::new(&device)?;
            rows8_projection_kernel.record_dispatch(
                &mut rows8_batch,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    fp16_weights.packed_storage(),
                    &targets_buffer,
                    &rows8_logits_grad,
                    &rows8_tile_stats,
                ],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(output_dim, 8), 1, 1],
            )?;
            rows8_stats_kernel.record_dispatch(
                &mut rows8_batch,
                &[&rows8_tile_stats, &targets_buffer, &rows8_stats],
                bytemuck::bytes_of(&linear_push),
                [rows as u32, 1, 1],
            )?;
            tape_input_grad_kernel.record_dispatch(
                &mut rows8_batch,
                &[
                    &rows8_logits_grad,
                    &rows8_stats,
                    fp16_weights.packed_storage(),
                    &rows8_grad_input,
                ],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(input_dim.div_ceil(2), 32), 1, 1],
            )?;
            tape_to_grad_kernel.record_dispatch(
                &mut rows8_batch,
                &[&rows8_stats, &rows8_logits_grad],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(rows * output_dim, 128), 1, 1],
            )?;
            rows8_batch.submit()?;

            let rows8_stats_values = rows8_stats.read_f32(rows * 5)?;
            for row in 0..rows {
                let base = row * 5;
                for offset in [0usize, 1, 3] {
                    let delta =
                        (packed_stats[base + offset] - rows8_stats_values[base + offset]).abs();
                    assert!(
                        delta <= 2.0e-6,
                        "fp16-ce-tape-rows8 row {row} stat {offset} drifted by {delta}"
                    );
                }
                assert_eq!(
                    packed_stats[base + 2].to_bits(),
                    rows8_stats_values[base + 2].to_bits(),
                    "fp16-ce-tape-rows8 row {row} target bits drifted"
                );
            }

            let rows8_input = rows8_grad_input.read_f32(rows * input_dim)?;
            let rows8_logits = rows8_logits_grad.read_f32(rows * output_dim)?;
            let rows8_input_max_abs = packed_input
                .iter()
                .zip(&rows8_input)
                .map(|(packed, rows8)| (packed - rows8).abs())
                .fold(0.0f32, f32::max);
            let rows8_logits_max_abs = packed_logits
                .iter()
                .zip(&rows8_logits)
                .map(|(packed, rows8)| (packed - rows8).abs())
                .fold(0.0f32, f32::max);
            assert!(
                rows8_input_max_abs <= 2.0e-6,
                "fp16-ce-tape-rows8 width-448 input gradient drifted from packed FP16 by {rows8_input_max_abs}"
            );
            assert!(
                rows8_logits_max_abs <= 2.0e-6,
                "fp16-ce-tape-rows8 width-448 CE adjoint drifted from packed FP16 by {rows8_logits_max_abs}"
            );
        }

        if input_dim <= 448
            && device.max_compute_shared_memory_bytes() >= 16_192
            && device.supports_compute_work_group_size_x(64)
            && device.supports_storage_buffer_bindings(9)
        {
            let rows16_stats = mean_ce_row_stats_buffer(&device, rows)?;
            let rows16_logits_grad = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
            let rows16_tile_stats =
                GpuBuffer::zeros_f32(&device, rows * output_dim.div_ceil(16) * 3)?;
            let rows16_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
            let rows16_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
            let rows16_fused_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
            let rows16_fused_grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
            let rows16_fused_partials = GpuBuffer::zeros_f32(
                &device,
                output_dim.div_ceil(CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_VOCAB_TILE)
                    * rows
                    * input_dim,
            )?;
            let rows16_projection_kernel = vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?;
            let rows16_stats_kernel = vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_ROWS16_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?;
            let mut rows16_batch = vulkan::ComputeBatch::new(&device)?;
            rows16_projection_kernel.record_dispatch(
                &mut rows16_batch,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    fp16_weights.packed_storage(),
                    &targets_buffer,
                    &rows16_logits_grad,
                    &rows16_tile_stats,
                ],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(output_dim, 16), 1, 1],
            )?;
            rows16_stats_kernel.record_dispatch(
                &mut rows16_batch,
                &[&rows16_tile_stats, &targets_buffer, &rows16_stats],
                bytemuck::bytes_of(&linear_push),
                [rows as u32, 1, 1],
            )?;
            tape_input_grad_kernel.record_dispatch(
                &mut rows16_batch,
                &[
                    &rows16_logits_grad,
                    &rows16_stats,
                    fp16_weights.packed_storage(),
                    &rows16_grad_input,
                ],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(input_dim.div_ceil(2), 32), 1, 1],
            )?;
            tape_weight_grad_kernel.record_dispatch(
                &mut rows16_batch,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    &rows16_logits_grad,
                    &rows16_stats,
                    &rows16_grad_weight,
                ],
                bytemuck::bytes_of(&lm_weight_grad_push),
                [
                    div_ceil_u32(
                        output_dim,
                        HierarchosLmWeightGradTopology::VocabRows8.vocab_rows_per_group() as usize,
                    ),
                    1,
                    1,
                ],
            )?;
            let fused_rows16_available = rows <= CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_MAX_ROWS
                && device.max_compute_shared_memory_bytes()
                    >= CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_SHARED_BYTES
                && device.supports_compute_work_group_size_x(64)
                && device.supports_storage_buffer_bindings(10);
            if fused_rows16_available {
                let fused_rows16_kernel = vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LmWeightGradPush>() as u32,
                )?;
                let fused_rows16_reduce_kernel = vulkan::ComputeKernel::new_with_access(
                    &device,
                    CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?;
                fused_rows16_kernel.record_dispatch(
                    &mut rows16_batch,
                    &[
                        &hidden,
                        &norm_weight,
                        &norm_bias,
                        &norm_mean,
                        &norm_rstd,
                        &rows16_logits_grad,
                        &rows16_stats,
                        fp16_weights.packed_storage(),
                        &rows16_fused_grad_weight,
                        &rows16_fused_partials,
                    ],
                    bytemuck::bytes_of(&lm_weight_grad_push),
                    [
                        div_ceil_u32(output_dim, CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_VOCAB_TILE),
                        1,
                        1,
                    ],
                )?;
                fused_rows16_reduce_kernel.record_dispatch(
                    &mut rows16_batch,
                    &[&rows16_fused_partials, &rows16_fused_grad_input],
                    bytemuck::bytes_of(&linear_push),
                    [div_ceil_u32(rows * input_dim, 64), 1, 1],
                )?;
            }
            tape_to_grad_kernel.record_dispatch(
                &mut rows16_batch,
                &[&rows16_stats, &rows16_logits_grad],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(rows * output_dim, 128), 1, 1],
            )?;
            rows16_batch.submit()?;

            let rows16_stats_values = rows16_stats.read_f32(rows * 5)?;
            for row in 0..rows {
                let base = row * 5;
                for offset in [0usize, 1, 3] {
                    let delta =
                        (packed_stats[base + offset] - rows16_stats_values[base + offset]).abs();
                    assert!(
                        delta <= 2.0e-6,
                        "fp16-ce-tape-rows16 row {row} stat {offset} drifted by {delta}"
                    );
                }
                assert_eq!(
                    packed_stats[base + 2].to_bits(),
                    rows16_stats_values[base + 2].to_bits(),
                    "fp16-ce-tape-rows16 row {row} target bits drifted"
                );
            }

            let rows16_input = rows16_grad_input.read_f32(rows * input_dim)?;
            let rows16_weight = rows16_grad_weight.read_f32(output_dim * input_dim)?;
            let rows16_logits = rows16_logits_grad.read_f32(rows * output_dim)?;
            let rows16_input_max_abs = packed_input
                .iter()
                .zip(&rows16_input)
                .map(|(packed, rows16)| (packed - rows16).abs())
                .fold(0.0f32, f32::max);
            let rows16_logits_max_abs = packed_logits
                .iter()
                .zip(&rows16_logits)
                .map(|(packed, rows16)| (packed - rows16).abs())
                .fold(0.0f32, f32::max);
            assert!(
                rows16_input_max_abs <= 2.0e-6,
                "fp16-ce-tape-rows16 width-448 input gradient drifted from packed FP16 by {rows16_input_max_abs}"
            );
            assert!(
                rows16_logits_max_abs <= 2.0e-6,
                "fp16-ce-tape-rows16 width-448 CE adjoint drifted from packed FP16 by {rows16_logits_max_abs}"
            );
            if fused_rows16_available {
                let fused_input = rows16_fused_grad_input.read_f32(rows * input_dim)?;
                let fused_weight = rows16_fused_grad_weight.read_f32(output_dim * input_dim)?;
                let fused_input_max_abs = rows16_input
                    .iter()
                    .zip(&fused_input)
                    .map(|(reference, fused)| (reference - fused).abs())
                    .fold(0.0f32, f32::max);
                let fused_weight_max_abs = rows16_weight
                    .iter()
                    .zip(&fused_weight)
                    .map(|(reference, fused)| (reference - fused).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    fused_input_max_abs <= 2.0e-6,
                    "fp16-ce-tape-rows16 fused W^T drifted from standalone tape W^T by {fused_input_max_abs}"
                );
                assert!(
                    fused_weight_max_abs <= 2.0e-6,
                    "fp16-ce-tape-rows16 fused dW drifted from standalone tape dW by {fused_weight_max_abs}"
                );
            }
        }

        let dw_variants = [
            (
                HierarchosLmWeightGradTopology::VocabRows4,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS4_SPV,
            ),
            (
                HierarchosLmWeightGradTopology::VocabRows8,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_SPV,
            ),
            (
                HierarchosLmWeightGradTopology::VocabRows16,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_STREAMING_FP16_PACKED_ROWS16_SPV,
            ),
        ];
        let mut dw_reference: Option<Vec<f32>> = None;
        let mut tested_dw = Vec::new();
        for (topology, shader) in dw_variants {
            let Some(kernel) = create_fp16_lm_weight_grad_kernel(&device, topology, shader)? else {
                continue;
            };
            let grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
            let mut batch = vulkan::ComputeBatch::new(&device)?;
            kernel.record_dispatch(
                &mut batch,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    &packed_grad_logits,
                    &grad_weight,
                ],
                bytemuck::bytes_of(&lm_weight_grad_push),
                [
                    div_ceil_u32(output_dim, topology.vocab_rows_per_group() as usize),
                    1,
                    1,
                ],
            )?;
            batch.submit()?;
            let values = grad_weight.read_f32(output_dim * input_dim)?;
            if let Some(reference) = dw_reference.as_ref() {
                let max_abs = reference
                    .iter()
                    .zip(&values)
                    .map(|(lhs, rhs)| (lhs - rhs).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 1.0e-7,
                    "{} width-448 dW drifted from the baseline topology by {max_abs}",
                    topology.label()
                );
            } else {
                dw_reference = Some(values);
            }
            tested_dw.push(topology);
        }
        assert!(tested_dw.contains(&HierarchosLmWeightGradTopology::VocabRows4));
        if device.supports_compute_work_group_size(
            HierarchosLmWeightGradTopology::VocabRows16.local_size(),
        ) {
            assert!(tested_dw.contains(&HierarchosLmWeightGradTopology::VocabRows16));
        }
        let tape_dw = tape_grad_weight.read_f32(output_dim * input_dim)?;
        let tape_dw_reference = dw_reference
            .as_ref()
            .context("width-448 packed dW reference topology was unavailable")?;
        let tape_dw_max_abs = tape_dw_reference
            .iter()
            .zip(&tape_dw)
            .map(|(packed, tape)| (packed - tape).abs())
            .fold(0.0f32, f32::max);
        assert!(
            tape_dw_max_abs <= 1.0e-7,
            "fp16-ce-tape direct-logit dW drifted from packed FP16 by {tape_dw_max_abs}"
        );

        // Exercise every topology variant of the direct-logit dW consumer.
        // Recreate an untouched logit tape because the compatibility oracle
        // above intentionally transformed the first tape after W^T/dW used it.
        let direct_dw_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let direct_dw_stats = mean_ce_row_stats_buffer(&device, rows)?;
        let mut direct_dw_setup = vulkan::ComputeBatch::new(&device)?;
        tape_row_stats_kernel.record_dispatch(
            &mut direct_dw_setup,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &norm_mean,
                &norm_rstd,
                fp16_weights.packed_storage(),
                &targets_buffer,
                &direct_dw_stats,
                &direct_dw_logits,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        direct_dw_setup.submit()?;

        let direct_dw_variants = [
            (
                HierarchosLmWeightGradTopology::VocabRows4,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS4_SPV,
            ),
            (
                HierarchosLmWeightGradTopology::VocabRows8,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_SPV,
            ),
            (
                HierarchosLmWeightGradTopology::VocabRows16,
                CROSS_ENTROPY_LINEAR_WEIGHT_GRAD_TAPE_FP16_PACKED_ROWS16_SPV,
            ),
        ];
        let mut tested_direct_dw = Vec::new();
        for (topology, shader) in direct_dw_variants {
            let Some(kernel) = create_fp16_lm_tape_weight_grad_kernel(&device, topology, shader)?
            else {
                continue;
            };
            let grad_weight = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
            let mut batch = vulkan::ComputeBatch::new(&device)?;
            kernel.record_dispatch(
                &mut batch,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    &direct_dw_logits,
                    &direct_dw_stats,
                    &grad_weight,
                ],
                bytemuck::bytes_of(&lm_weight_grad_push),
                [
                    div_ceil_u32(output_dim, topology.vocab_rows_per_group() as usize),
                    1,
                    1,
                ],
            )?;
            batch.submit()?;
            let values = grad_weight.read_f32(output_dim * input_dim)?;
            let max_abs = tape_dw_reference
                .iter()
                .zip(&values)
                .map(|(packed, direct)| (packed - direct).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 1.0e-7,
                "{} direct-logit dW drifted from packed FP16 by {max_abs}",
                topology.label()
            );
            tested_direct_dw.push(topology);
        }
        assert!(tested_direct_dw.contains(&HierarchosLmWeightGradTopology::VocabRows4));
        assert!(tested_direct_dw.contains(&HierarchosLmWeightGradTopology::VocabRows8));
        if device.supports_compute_work_group_size(
            HierarchosLmWeightGradTopology::VocabRows16.local_size(),
        ) {
            assert!(tested_direct_dw.contains(&HierarchosLmWeightGradTopology::VocabRows16));
        }

        let variants = [
            (
                HierarchosLmExecutionArm::Fp16Native,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_SPV,
            ),
            (
                HierarchosLmExecutionArm::Fp16NativeReuse64,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE64_SPV,
            ),
            (
                HierarchosLmExecutionArm::Fp16NativeReuse128,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE128_SPV,
            ),
            (
                HierarchosLmExecutionArm::Fp16NativeReuse224,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_STREAMING_FP16_NATIVE_REUSE224_SPV,
            ),
        ];
        let mut tested = Vec::new();
        for (arm, shader) in variants {
            let Some(kernel) = create_native_fp16_lm_input_grad_kernel(&device, arm, shader)?
            else {
                continue;
            };
            let native_grad_input = GpuBuffer::zeros_f32(&device, rows * input_dim)?;
            let native_grad_logits = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
            let mut native_batch = vulkan::ComputeBatch::new(&device)?;
            kernel.record_dispatch(
                &mut native_batch,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    fp16_weights.packed_storage(),
                    &stats,
                    &native_grad_input,
                    &native_grad_logits,
                ],
                bytemuck::bytes_of(&linear_push),
                [rows as u32, 1, 1],
            )?;
            native_batch.submit()?;

            let native_input = native_grad_input.read_f32(rows * input_dim)?;
            let native_logits = native_grad_logits.read_f32(rows * output_dim)?;
            let input_max_abs = packed_input
                .iter()
                .zip(&native_input)
                .map(|(packed, native)| (packed - native).abs())
                .fold(0.0f32, f32::max);
            let logits_max_abs = packed_logits
                .iter()
                .zip(&native_logits)
                .map(|(packed, native)| (packed - native).abs())
                .fold(0.0f32, f32::max);
            assert!(
                input_max_abs <= 1.0e-7,
                "{} width-448 input gradient drifted from packed FP16 by {input_max_abs}",
                arm.label()
            );
            assert!(
                logits_max_abs <= 1.0e-7,
                "{} width-448 CE adjoint drifted from packed FP16 by {logits_max_abs}",
                arm.label()
            );
            tested.push(arm);
        }

        assert!(tested.contains(&HierarchosLmExecutionArm::Fp16Native));
        if device.max_compute_shared_memory_bytes()
            >= HierarchosLmExecutionArm::Fp16NativeReuse224
                .native_fp16_shared_memory_bytes()
                .expect("reuse224 shared-memory size")
        {
            assert!(tested.contains(&HierarchosLmExecutionArm::Fp16NativeReuse224));
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU fused-adjoint dX tile-width microprofile; run explicitly on the target Vulkan device"]
    fn lm_fused_adjoint_dx_tile_microprofile() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let rows = std::env::var("HIERARCHOS_VULKAN_LM_DX_TILE_PROFILE_ROWS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_MAX_ROWS);
        let input_dim = 448usize;
        let output_dim = std::env::var("HIERARCHOS_VULKAN_LM_DX_TILE_PROFILE_VOCAB_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(50_257);
        if rows == 0 || rows > CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_MAX_ROWS || output_dim == 0 {
            bail!(
                "LM dX tile microprofile rows must be 1..={} and vocabulary size must be positive",
                CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_MAX_ROWS
            );
        }
        if device.max_compute_shared_memory_bytes()
            < CROSS_ENTROPY_LINEAR_FUSED_ADJOINT_SHARED_BYTES
            || !device.supports_compute_work_group_size_x(64)
            || !device.supports_compute_work_group_size_x(128)
            || !device.supports_storage_buffer_bindings(10)
        {
            eprintln!(
                "Hierarchos LM dX tile microprofile skipped: device={} shared={}B",
                device.name(),
                device.max_compute_shared_memory_bytes()
            );
            return Ok(());
        }

        let weight_values = (0..output_dim * input_dim)
            .map(|index| ((index % 127) as f32 - 63.0) * (1.0 / 4096.0))
            .collect::<Vec<_>>();
        let norm_weight = (0..input_dim)
            .map(|index| 0.95 + (index % 17) as f32 * 0.002)
            .collect::<Vec<_>>();
        let norm_bias = (0..input_dim)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.0005)
            .collect::<Vec<_>>();
        let trainer = HierarchosOutNormHeadTrainer::new(
            device.clone(),
            input_dim,
            output_dim,
            rows,
            &weight_values,
            &norm_weight,
            &norm_bias,
        )?;

        let shared = trainer.shared_lm_head();
        let mirror = mixed_precision::VulkanParameterStorageMirror::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
            weight_values.len(),
        )?;
        let refresher = mixed_precision::VulkanParameterStorageMirrorRefresher::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
        )?;
        let mut refresh = vulkan::ComputeBatch::new(&device)?;
        refresher.record_refresh(&mut refresh, shared.weight_buffer(), &mirror)?;
        refresh.submit()?;
        shared.install_fp16_parameter_storage_mirror(mirror)?;
        let fp16_weight = shared
            .fp16_parameter_storage_mirror()?
            .context("LM dX tile microprofile failed to install FP16 mirror")?;

        let synthetic_hidden = (0..rows * input_dim)
            .map(|index| ((index % 61) as f32 - 30.0) * (1.0 / 64.0))
            .collect::<Vec<_>>();
        let synthetic_targets = (0..rows)
            .map(|row| ((row.wrapping_mul(7919).wrapping_add(17)) % output_dim) as u32)
            .collect::<Vec<_>>();
        let mut setup = vulkan::ComputeBatch::new(&device)?;
        setup.upload_f32(&trainer.input_hidden, &synthetic_hidden)?;
        setup.upload_u32(&trainer.targets, &synthetic_targets)?;
        let linear_push =
            trainer.record_norm_head_stats(&mut setup, &trainer.input_hidden, rows)?;
        setup.submit()?;

        // Freeze the FP32 logit tape and compact CE row stats once. The timing
        // below therefore measures only fused dW+dX emission plus deterministic
        // dX reduction, which is the seam this experiment is meant to isolate.
        let (projection_kernel, stats_kernel, forward_vocab_tile) = trainer
            .fp16_ce_cross_row_kernels(HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints)?;
        let mut forward = vulkan::ComputeBatch::new(&device)?;
        projection_kernel.record_dispatch(
            &mut forward,
            &[
                &trainer.input_hidden,
                &trainer.norm_weight,
                &trainer.norm_bias,
                &trainer.norm_mean,
                &trainer.norm_rstd,
                fp16_weight.packed_storage(),
                &trainer.targets,
                &trainer.ce_grad_tape,
                &trainer.ce_tile_stats,
            ],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(output_dim, forward_vocab_tile), 1, 1],
        )?;
        stats_kernel.record_dispatch(
            &mut forward,
            &[
                &trainer.ce_tile_stats,
                &trainer.targets,
                &trainer.ce_row_stats,
            ],
            bytemuck::bytes_of(&linear_push),
            [rows as u32, 1, 1],
        )?;
        forward.submit()?;

        let weight_grad_push = LmWeightGradPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            accumulate: 0,
            z_loss_weight: 0.0,
            activation_clamp: linear_push.activation_clamp,
        };
        let fused_access = [
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadWrite,
            vulkan::BindingAccess::MayWrite,
        ];
        let reduce_access = [
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
        ];
        let variants = [
            (
                "tile64-wg64",
                64usize,
                64usize,
                CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_SPV,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_SPV,
            ),
            (
                "tile128-wg64",
                128usize,
                64usize,
                CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE128_SPV,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE128_SPV,
            ),
            (
                "tile128-wg128",
                128usize,
                128usize,
                CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE128_WG128_SPV,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE128_SPV,
            ),
            (
                "tile256-wg64",
                256usize,
                64usize,
                CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_SPV,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE256_SPV,
            ),
            (
                "tile256-wg128",
                256usize,
                128usize,
                CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_WG128_SPV,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE256_SPV,
            ),
            (
                "tile256-wg256",
                256usize,
                256usize,
                CROSS_ENTROPY_LINEAR_ADJOINTS_TAPE_FP16_PACKED_FUSED_PRIVATE_HIDDEN_TILE256_WG256_SPV,
                CROSS_ENTROPY_LINEAR_INPUT_GRAD_TILE_REDUCE_TILE256_SPV,
            ),
        ];
        let mut reference_dx: Option<Vec<f32>> = None;
        let mut measured = Vec::with_capacity(variants.len());

        for (label, vocab_tile, workgroup_size, fused_spv, reduce_spv) in variants {
            if !device.supports_compute_work_group_size_x(workgroup_size as u32) {
                eprintln!("  {label} skipped: workgroup size {workgroup_size} is unavailable");
                continue;
            }
            let partial_len = output_dim
                .div_ceil(vocab_tile)
                .checked_mul(rows)
                .and_then(|count| count.checked_mul(input_dim))
                .context("LM dX tile microprofile partial count overflow")?;
            let partials = GpuBuffer::zeros_f32(&device, partial_len)?;
            let fused_kernel = vulkan::ComputeKernel::new_with_access(
                &device,
                fused_spv,
                &fused_access,
                std::mem::size_of::<LmWeightGradPush>() as u32,
            )?;
            let reduce_kernel = vulkan::ComputeKernel::new_with_access(
                &device,
                reduce_spv,
                &reduce_access,
                std::mem::size_of::<LinearPush>() as u32,
            )?;

            let record = |commands: &mut vulkan::ComputeBatch| -> Result<()> {
                fused_kernel.record_dispatch(
                    commands,
                    &[
                        &trainer.input_hidden,
                        &trainer.norm_weight,
                        &trainer.norm_bias,
                        &trainer.norm_mean,
                        &trainer.norm_rstd,
                        &trainer.ce_grad_tape,
                        &trainer.ce_row_stats,
                        fp16_weight.packed_storage(),
                        &trainer.grad_lm_weight,
                        &partials,
                    ],
                    bytemuck::bytes_of(&weight_grad_push),
                    [div_ceil_u32(output_dim, vocab_tile), 1, 1],
                )?;
                reduce_kernel.record_dispatch(
                    commands,
                    &[&partials, &trainer.grad_norm_hidden],
                    bytemuck::bytes_of(&linear_push),
                    [div_ceil_u32(rows * input_dim, 64), 1, 1],
                )?;
                Ok(())
            };

            let mut correctness = vulkan::ComputeBatch::new(&device)?;
            record(&mut correctness)?;
            correctness.submit()?;
            let dx = trainer.grad_norm_hidden.read_f32(rows * input_dim)?;
            let dx_max_abs = if let Some(reference) = reference_dx.as_ref() {
                reference
                    .iter()
                    .zip(&dx)
                    .map(|(baseline, candidate)| (baseline - candidate).abs())
                    .fold(0.0f32, f32::max)
            } else {
                reference_dx = Some(dx);
                0.0
            };
            assert!(
                dx_max_abs <= 2.0e-6,
                "{label} fused-adjoint dX drifted from deterministic tile64 by {dx_max_abs}"
            );

            let mut warmup = vulkan::ComputeBatch::new(&device)?;
            record(&mut warmup)?;
            warmup.submit()?;
            let mut samples = Vec::with_capacity(5);
            for _ in 0..5 {
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                record(&mut commands)?;
                let started = std::time::Instant::now();
                commands.submit()?;
                samples.push(started.elapsed().as_secs_f64() * 1_000.0);
            }
            samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
            let median_ms = samples[samples.len() / 2];
            let partial_bytes = partial_len
                .checked_mul(std::mem::size_of::<f32>())
                .context("LM dX tile microprofile byte count overflow")?;
            eprintln!(
                "  {label} fused+reduce={median_ms:.4}ms partials={partial_bytes}B ({:.3} MiB) dx_max_abs={dx_max_abs:.9e}",
                partial_bytes as f64 / (1024.0 * 1024.0)
            );
            measured.push((label, vocab_tile, median_ms, partial_bytes, dx_max_abs));
        }

        let baseline = measured[0];
        eprintln!(
            "Hierarchos LM fused-adjoint dX tile microprofile device={} subgroup={} rows={} vocab={} baseline={} {:.4}ms {:.3}MiB",
            device.name(),
            device.subgroup_capabilities().subgroup_size,
            rows,
            output_dim,
            baseline.0,
            baseline.2,
            baseline.3 as f64 / (1024.0 * 1024.0)
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU LM width-448 backward microprofile; run explicitly on the target Vulkan device"]
    fn lm_width448_backward_topology_microprofile() -> Result<()> {
        let Some(device) = microprofile_device()? else {
            return Ok(());
        };
        let rows = std::env::var("HIERARCHOS_VULKAN_LM_MICROPROFILE_ROWS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(2);
        let input_dim = 448usize;
        let output_dim = std::env::var("HIERARCHOS_VULKAN_LM_MICROPROFILE_VOCAB_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(50_257);
        if rows == 0 || output_dim == 0 {
            bail!("LM microprofile rows and vocabulary size must be positive");
        }

        let weight_values = (0..output_dim * input_dim)
            .map(|index| ((index % 127) as f32 - 63.0) * (1.0 / 4096.0))
            .collect::<Vec<_>>();
        let norm_weight = (0..input_dim)
            .map(|index| 0.95 + (index % 17) as f32 * 0.002)
            .collect::<Vec<_>>();
        let norm_bias = (0..input_dim)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.0005)
            .collect::<Vec<_>>();
        let mut trainer = HierarchosOutNormHeadTrainer::new(
            device.clone(),
            input_dim,
            output_dim,
            rows,
            &weight_values,
            &norm_weight,
            &norm_bias,
        )?;

        let shared = trainer.shared_lm_head();
        let mirror = mixed_precision::VulkanParameterStorageMirror::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
            weight_values.len(),
        )?;
        let refresher = mixed_precision::VulkanParameterStorageMirrorRefresher::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
        )?;
        let mut refresh = vulkan::ComputeBatch::new(&device)?;
        refresher.record_refresh(&mut refresh, shared.weight_buffer(), &mirror)?;
        refresh.submit()?;
        shared.install_fp16_parameter_storage_mirror(mirror)?;
        let fp16_weight = shared
            .fp16_parameter_storage_mirror()?
            .context("LM microprofile failed to install FP16 mirror")?;

        let synthetic_hidden = (0..rows * input_dim)
            .map(|index| ((index % 61) as f32 - 30.0) * (1.0 / 64.0))
            .collect::<Vec<_>>();
        let synthetic_targets = (0..rows)
            .map(|row| ((row.wrapping_mul(7919).wrapping_add(17)) % output_dim) as u32)
            .collect::<Vec<_>>();
        let mut setup = vulkan::ComputeBatch::new(&device)?;
        setup.upload_f32(&trainer.input_hidden, &synthetic_hidden)?;
        setup.upload_u32(&trainer.targets, &synthetic_targets)?;
        let linear_push =
            trainer.record_norm_head_stats(&mut setup, &trainer.input_hidden, rows)?;
        setup.submit()?;

        let input_arms = [
            HierarchosLmExecutionArm::Fp16Packed,
            HierarchosLmExecutionArm::Fp16CeTape,
            HierarchosLmExecutionArm::Fp16CeTapeRows8,
            HierarchosLmExecutionArm::Fp16CeTapeRows16,
            HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4,
            HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints,
            HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints,
            HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints,
            HierarchosLmExecutionArm::Fp16Native,
            HierarchosLmExecutionArm::Fp16NativeReuse64,
            HierarchosLmExecutionArm::Fp16NativeReuse128,
            HierarchosLmExecutionArm::Fp16NativeReuse224,
        ];
        let dw_topologies = [
            HierarchosLmWeightGradTopology::VocabRows4,
            HierarchosLmWeightGradTopology::VocabRows8,
            HierarchosLmWeightGradTopology::VocabRows16,
        ];
        eprintln!(
            "Hierarchos LM width448 microprofile device_index={} device={} subgroup={} shared={}B rows={} vocab={}",
            device.physical_device_index(),
            device.name(),
            device.subgroup_capabilities().subgroup_size,
            device.max_compute_shared_memory_bytes(),
            rows,
            output_dim
        );
        let mut measured = 0usize;
        for input_grad_arm in input_arms {
            if input_grad_arm.fuses_ce_adjoints()
                && (trainer
                    .cross_entropy_linear_adjoints_tape_fp16_packed_fused
                    .is_none()
                    || trainer.ce_input_grad_partials.is_none())
            {
                continue;
            }
            if matches!(
                input_grad_arm,
                HierarchosLmExecutionArm::Fp16CeTapeRows8
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints
            ) && trainer.fp16_ce_cross_row_kernels(input_grad_arm).is_err()
            {
                continue;
            }
            if !matches!(
                input_grad_arm,
                HierarchosLmExecutionArm::Fp16CeTape
                    | HierarchosLmExecutionArm::Fp16CeTapeRows8
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints
                    | HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints
            ) && trainer.fp16_lm_input_grad_kernel(input_grad_arm).is_err()
            {
                continue;
            }
            for weight_grad_topology in dw_topologies {
                if input_grad_arm.fuses_ce_adjoints()
                    && weight_grad_topology != HierarchosLmWeightGradTopology::VocabRows8
                {
                    continue;
                }
                if !input_grad_arm.fuses_ce_adjoints()
                    && trainer
                        .fp16_lm_weight_grad_kernel(weight_grad_topology)
                        .is_err()
                {
                    continue;
                }
                let fused_topologies: &[_] = if input_grad_arm.fuses_ce_adjoints() {
                    &[
                        lm_execution::HierarchosLmFusedAdjointTopology::SharedHidden,
                        lm_execution::HierarchosLmFusedAdjointTopology::PrivateHidden,
                        lm_execution::HierarchosLmFusedAdjointTopology::PrivateHiddenTile256,
                    ]
                } else {
                    &[lm_execution::HierarchosLmFusedAdjointTopology::SharedHidden]
                };
                for &fused_adjoint_topology in fused_topologies {
                    if input_grad_arm.fuses_ce_adjoints()
                        && trainer
                            .fp16_fused_adjoint_kernel(fused_adjoint_topology)
                            .is_err()
                    {
                        continue;
                    }
                    let plan = lm_execution::HierarchosLmBackwardPlan {
                        input_grad_arm,
                        weight_grad_topology,
                        fused_adjoint_topology,
                    };
                    let mut samples = Vec::with_capacity(5);
                    for _ in 0..5 {
                        samples.push(trainer.time_fp16_lm_backward_plan_ms(
                            &fp16_weight,
                            rows,
                            linear_push,
                            plan,
                        )?);
                    }
                    samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
                    let median_ms = samples[samples.len() / 2];
                    if input_grad_arm.fuses_ce_adjoints() {
                        eprintln!(
                            "  {}+{}+{}={median_ms:.4}ms",
                            input_grad_arm.label(),
                            weight_grad_topology.label(),
                            fused_adjoint_topology.label()
                        );
                    } else {
                        eprintln!(
                            "  {}+{}={median_ms:.4}ms",
                            input_grad_arm.label(),
                            weight_grad_topology.label()
                        );
                    }
                    measured += 1;
                }
            }
        }
        assert!(
            measured > 0,
            "LM width-448 microprofile found no runnable plans"
        );
        trainer.configure_fp16_lm_execution_arm()?;
        eprintln!(
            "Hierarchos LM width448 selector selected={}+{}+{}",
            trainer.lm_execution_arm().label(),
            trainer
                .lm_weight_grad_topology()
                .expect("FP16 LM microprofile selector must install a dW topology")
                .label(),
            trainer.lm_fused_adjoint_topology.label()
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU rows16 LM projection->stats seam microprofile; run explicitly on the target Vulkan device"]
    fn lm_rows16_forward_stats_seam_microprofile() -> Result<()> {
        let Some(device) = microprofile_device()? else {
            return Ok(());
        };
        let rows = std::env::var("HIERARCHOS_VULKAN_LM_SEAM_PROFILE_ROWS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(16);
        let output_dim = std::env::var("HIERARCHOS_VULKAN_LM_SEAM_PROFILE_VOCAB_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(50_257);
        let repetitions = std::env::var("HIERARCHOS_VULKAN_LM_SEAM_PROFILE_REPETITIONS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(32);
        let input_dim = 448usize;
        if rows == 0 || output_dim == 0 || repetitions == 0 {
            bail!("LM rows16 seam profile rows, vocabulary size, and repetitions must be positive");
        }
        if device.max_compute_shared_memory_bytes() < 16_384
            || !device.supports_compute_work_group_size_x(64)
            || !device.supports_storage_buffer_bindings(9)
        {
            eprintln!(
                "Hierarchos LM rows16 seam microprofile skipped: device_index={} device={} shared={}B",
                device.physical_device_index(),
                device.name(),
                device.max_compute_shared_memory_bytes()
            );
            return Ok(());
        }

        let weight_values = (0..output_dim * input_dim)
            .map(|index| ((index % 127) as f32 - 63.0) * (1.0 / 4096.0))
            .collect::<Vec<_>>();
        let master_weight = GpuBuffer::from_f32(&device, &weight_values)?;
        let fp16_weights = mixed_precision::VulkanParameterStorageMirror::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
            weight_values.len(),
        )?;
        let refresher = mixed_precision::VulkanParameterStorageMirrorRefresher::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
        )?;
        let mut refresh = vulkan::ComputeBatch::new(&device)?;
        refresher.record_refresh(&mut refresh, &master_weight, &fp16_weights)?;
        refresh.submit()?;

        let hidden_values = (0..rows * input_dim)
            .map(|index| ((index % 61) as f32 - 30.0) * (1.0 / 64.0))
            .collect::<Vec<_>>();
        let norm_weight_values = (0..input_dim)
            .map(|index| 0.95 + (index % 17) as f32 * 0.002)
            .collect::<Vec<_>>();
        let norm_bias_values = (0..input_dim)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.0005)
            .collect::<Vec<_>>();
        let targets = (0..rows)
            .map(|row| ((row.wrapping_mul(7919).wrapping_add(17)) % output_dim) as u32)
            .collect::<Vec<_>>();
        let hidden = GpuBuffer::from_f32(&device, &hidden_values)?;
        let norm_weight = GpuBuffer::from_f32(&device, &norm_weight_values)?;
        let norm_bias = GpuBuffer::from_f32(&device, &norm_bias_values)?;
        let norm_mean = GpuBuffer::from_f32(&device, &vec![0.0; rows])?;
        let norm_rstd = GpuBuffer::from_f32(&device, &vec![1.0; rows])?;
        let targets_buffer = GpuBuffer::from_u32(&device, &targets)?;
        let logit_tape = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let dot4_logit_tape = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let cluster4_logit_tape = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let tile_count = output_dim.div_ceil(16);
        let tile_stats = GpuBuffer::zeros_f32(&device, rows * tile_count * 3)?;
        let dot4_tile_stats = GpuBuffer::zeros_f32(&device, rows * tile_count * 3)?;
        let cluster4_tile_stats = GpuBuffer::zeros_f32(&device, rows * tile_count * 3)?;
        let row_stats = mean_ce_row_stats_buffer(&device, rows)?;
        let dot4_row_stats = mean_ce_row_stats_buffer(&device, rows)?;
        let cluster4_row_stats = mean_ce_row_stats_buffer(&device, rows)?;

        let projection = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let dot4_projection = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_DOT4_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let cluster4_projection = if device.supports_compute_subgroup_clustered_arithmetic() {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                CROSS_ENTROPY_LINEAR_LOGIT_TAPE_FP16_PACKED_ROWS16_CLUSTER4_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?)
        } else {
            None
        };
        let stats = vulkan::ComputeKernel::new_with_access(
            &device,
            CROSS_ENTROPY_ROW_STATS_TILE_PARTIALS_ROWS16_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LinearPush>() as u32,
        )?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            z_loss_weight: 0.0,
            activation_clamp: f32::MAX,
        };

        let mut correctness = vulkan::ComputeBatch::new(&device)?;
        projection.record_dispatch(
            &mut correctness,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &norm_mean,
                &norm_rstd,
                fp16_weights.packed_storage(),
                &targets_buffer,
                &logit_tape,
                &tile_stats,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(output_dim, 16), 1, 1],
        )?;
        stats.record_dispatch(
            &mut correctness,
            &[&tile_stats, &targets_buffer, &row_stats],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        dot4_projection.record_dispatch(
            &mut correctness,
            &[
                &hidden,
                &norm_weight,
                &norm_bias,
                &norm_mean,
                &norm_rstd,
                fp16_weights.packed_storage(),
                &targets_buffer,
                &dot4_logit_tape,
                &dot4_tile_stats,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(output_dim, 16), 1, 1],
        )?;
        stats.record_dispatch(
            &mut correctness,
            &[&dot4_tile_stats, &targets_buffer, &dot4_row_stats],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        if let Some(cluster4_projection) = cluster4_projection.as_ref() {
            cluster4_projection.record_dispatch(
                &mut correctness,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    fp16_weights.packed_storage(),
                    &targets_buffer,
                    &cluster4_logit_tape,
                    &cluster4_tile_stats,
                ],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(output_dim, 16), 1, 1],
            )?;
            stats.record_dispatch(
                &mut correctness,
                &[&cluster4_tile_stats, &targets_buffer, &cluster4_row_stats],
                bytemuck::bytes_of(&push),
                [rows as u32, 1, 1],
            )?;
        }
        correctness.submit()?;
        let baseline_logits = logit_tape.read_f32(rows * output_dim)?;
        let dot4_logits = dot4_logit_tape.read_f32(rows * output_dim)?;
        let max_logit_abs = baseline_logits
            .iter()
            .zip(&dot4_logits)
            .map(|(baseline, dot4)| (baseline - dot4).abs())
            .fold(0.0f32, f32::max);
        let baseline_stats = row_stats.read_f32(rows * 5)?;
        let dot4_stats_values = dot4_row_stats.read_f32(rows * 5)?;
        let max_stats_abs = baseline_stats
            .iter()
            .zip(&dot4_stats_values)
            .map(|(baseline, dot4)| (baseline - dot4).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_logit_abs <= 1.0e-4,
            "rows16 dot4 logit reduction drifted from serial rows16 by {max_logit_abs}"
        );
        assert!(
            max_stats_abs <= 1.0e-4,
            "rows16 dot4 CE stats drifted from serial rows16 by {max_stats_abs}"
        );
        let cluster4_max_abs = if cluster4_projection.is_some() {
            let cluster4_logits = cluster4_logit_tape.read_f32(rows * output_dim)?;
            let cluster4_stats_values = cluster4_row_stats.read_f32(rows * 5)?;
            let max_logit_abs = baseline_logits
                .iter()
                .zip(&cluster4_logits)
                .map(|(baseline, cluster4)| (baseline - cluster4).abs())
                .fold(0.0f32, f32::max);
            let max_stats_abs = baseline_stats
                .iter()
                .zip(&cluster4_stats_values)
                .map(|(baseline, cluster4)| (baseline - cluster4).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_logit_abs <= 1.0e-4,
                "rows16 cluster4 logit reduction drifted from serial rows16 by {max_logit_abs}"
            );
            assert!(
                max_stats_abs <= 1.0e-4,
                "rows16 cluster4 CE stats drifted from serial rows16 by {max_stats_abs}"
            );
            Some((max_logit_abs, max_stats_abs))
        } else {
            None
        };

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        for _ in 0..repetitions {
            projection.record_dispatch(
                &mut commands,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    fp16_weights.packed_storage(),
                    &targets_buffer,
                    &logit_tape,
                    &tile_stats,
                ],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(output_dim, 16), 1, 1],
            )?;
            stats.record_dispatch(
                &mut commands,
                &[&tile_stats, &targets_buffer, &row_stats],
                bytemuck::bytes_of(&push),
                [rows as u32, 1, 1],
            )?;
        }
        let started = std::time::Instant::now();
        commands.submit()?;
        let baseline_pair_ms = started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64;

        let mut dot4_commands = vulkan::ComputeBatch::new(&device)?;
        for _ in 0..repetitions {
            dot4_projection.record_dispatch(
                &mut dot4_commands,
                &[
                    &hidden,
                    &norm_weight,
                    &norm_bias,
                    &norm_mean,
                    &norm_rstd,
                    fp16_weights.packed_storage(),
                    &targets_buffer,
                    &dot4_logit_tape,
                    &dot4_tile_stats,
                ],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(output_dim, 16), 1, 1],
            )?;
            stats.record_dispatch(
                &mut dot4_commands,
                &[&dot4_tile_stats, &targets_buffer, &dot4_row_stats],
                bytemuck::bytes_of(&push),
                [rows as u32, 1, 1],
            )?;
        }
        let dot4_started = std::time::Instant::now();
        dot4_commands.submit()?;
        let dot4_pair_ms = dot4_started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64;
        eprintln!(
            "Hierarchos LM rows16 forward->stats seam device_index={} device={} rows={} vocab={} tile_partials={}B repetitions={} serial_host_pair_ms={baseline_pair_ms:.6} dot4_host_pair_ms={dot4_pair_ms:.6} logit_max_abs={max_logit_abs:.9e} stats_max_abs={max_stats_abs:.9e}; set HIERARCHOS_VULKAN_PROFILE_KERNELS=1 for per-dispatch GPU timestamps",
            device.physical_device_index(),
            device.name(),
            rows,
            output_dim,
            rows * tile_count * 3 * std::mem::size_of::<f32>(),
            repetitions,
        );
        if let Some((cluster4_logit_max_abs, cluster4_stats_max_abs)) = cluster4_max_abs {
            eprintln!(
                "Hierarchos LM rows16 cluster4 correctness device_index={} device={} logit_max_abs={cluster4_logit_max_abs:.9e} stats_max_abs={cluster4_stats_max_abs:.9e}",
                device.physical_device_index(),
                device.name(),
            );
        }
        Ok(())
    }

    #[test]
    fn tied_embedding_and_lm_loss_share_optimizer_identity() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let shared =
            SharedLmHeadParameter::new(device, 2, 3, &[0.10, -0.20, 0.30, 0.40, -0.50, 0.60])?;
        let tied = TiedTokenEmbeddingOp::from_shared_parameter(shared.clone(), 2)?;
        let mut head = HierarchosHeadTrainer::from_shared_lm_head(shared.clone(), 2)?;
        assert!(shared.shares_identity_with(&tied.shared_parameter()));
        assert!(shared.shares_identity_with(&head.shared_lm_head()));

        let result = head.train_step(
            &[0.25, -0.75],
            &[1],
            AdamWHyperParams {
                lr: 1.0e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1.0e-8,
                weight_decay: 0.01,
            },
        )?;
        assert_eq!(result.step, 1);
        assert_eq!(shared.step(), 1);
        assert!(shared.shares_identity_with(&head.shared_lm_head()));
        Ok(())
    }

    #[test]
    fn phased_shared_lm_accumulation_matches_one_combined_update() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let initial = [0.10, -0.20, 0.30, 0.40, -0.50, 0.60];
        let norm_weight = [1.0, 0.75];
        let norm_bias = [0.05, -0.10];
        let hidden = [0.25, -0.75, 0.60, 0.20];
        let targets = [1, 2];
        let h_ids = [0, 2];
        let h_grad = [0.02, -0.03, 0.04, 0.01];
        let l_ids = [1, 2];
        let l_grad = [-0.05, 0.02, 0.03, -0.04];
        let combined_ids = [0, 2, 1, 2];
        let combined_grad = [0.02, -0.03, 0.04, 0.01, -0.05, 0.02, 0.03, -0.04];
        let hyper = AdamWHyperParams {
            lr: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.01,
        };

        let reference_shared = SharedLmHeadParameter::new(device.clone(), 2, 3, &initial)?;
        let mut reference = HierarchosOutNormHeadTrainer::from_shared_lm_head(
            reference_shared.clone(),
            4,
            &norm_weight,
            &norm_bias,
        )?;
        let reference_step = reference.train_step_with_tied_embedding_grad(
            &hidden,
            &targets,
            &combined_ids,
            &combined_grad,
            hyper,
        )?;

        let phased_shared = SharedLmHeadParameter::new(device.clone(), 2, 3, &initial)?;
        let h_embedding = TiedTokenEmbeddingOp::from_shared_parameter(phased_shared.clone(), 2)?;
        let l_embedding = TiedTokenEmbeddingOp::from_shared_parameter(phased_shared.clone(), 2)?;
        let h_id_buffer = GpuBuffer::from_u32(&device, &h_ids)?;
        let h_grad_buffer = GpuBuffer::from_f32(&device, &h_grad)?;
        let l_id_buffer = GpuBuffer::from_u32(&device, &l_ids)?;
        let l_grad_buffer = GpuBuffer::from_f32(&device, &l_grad)?;
        let mut accumulation = vulkan::ComputeBatch::new(&device)?;
        phased_shared.record_zero_grad(&mut accumulation)?;
        h_embedding.record_backward_accumulate(
            &mut accumulation,
            h_ids.len(),
            &h_id_buffer,
            &h_grad_buffer,
        )?;
        l_embedding.record_backward_accumulate(
            &mut accumulation,
            l_ids.len(),
            &l_id_buffer,
            &l_grad_buffer,
        )?;
        accumulation.submit()?;

        let mut phased = HierarchosOutNormHeadTrainer::from_shared_lm_head(
            phased_shared.clone(),
            4,
            &norm_weight,
            &norm_bias,
        )?;
        let phased_step = phased.train_step_finalize_shared_lm(&hidden, &targets, hyper)?;

        assert_eq!(reference_step.step, 1);
        assert_eq!(phased_step.step, 1);
        assert_eq!(reference_shared.step(), phased_shared.step());
        let reference_weight = reference_shared.weights()?;
        let phased_weight = phased_shared.weights()?;
        let max_abs = reference_weight
            .iter()
            .zip(&phased_weight)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 2.0e-6,
            "phased/shared LM update drifted by {max_abs}"
        );
        Ok(())
    }

    #[test]
    fn out_norm_large_tied_embedding_radix_matches_dense_sparse_reference() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let context_dim = 3usize;
        let vocab_size = 97usize;
        let token_count = EMBEDDING_SEGMENTED_SORT_CAPACITY + 513;
        let lm_weight = (0..vocab_size * context_dim)
            .map(|index| ((index % 29) as f32 - 14.0) * (1.0 / 256.0))
            .collect::<Vec<_>>();
        let norm_weight = [1.0f32, 0.85, 1.15];
        let norm_bias = [0.025f32, -0.05, 0.075];
        let hidden = [0.25f32, -0.75, 0.40, -0.35, 0.60, 0.15];
        let targets = [11u32, 73];
        let token_ids = (0..token_count)
            .map(|index| ((index * 37 + index / 11) % vocab_size) as u32)
            .collect::<Vec<_>>();
        let embedding_grad = (0..token_count * context_dim)
            .map(|index| ((index % 31) as f32 - 15.0) * (1.0 / 8192.0))
            .collect::<Vec<_>>();
        let mut dense_sparse_grad = vec![0.0f32; vocab_size * context_dim];
        for (position, &token) in token_ids.iter().enumerate() {
            for col in 0..context_dim {
                dense_sparse_grad[token as usize * context_dim + col] +=
                    embedding_grad[position * context_dim + col];
            }
        }
        let hyper = AdamWHyperParams {
            lr: 3.0e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.01,
        };

        let radix_shared =
            SharedLmHeadParameter::new(device.clone(), context_dim, vocab_size, &lm_weight)?;
        let mut radix = HierarchosOutNormHeadTrainer::from_shared_lm_head(
            radix_shared.clone(),
            token_count,
            &norm_weight,
            &norm_bias,
        )?;
        let radix_step = radix.train_step_with_tied_embedding_grad(
            &hidden,
            &targets,
            &token_ids,
            &embedding_grad,
            hyper,
        )?;

        let reference_shared =
            SharedLmHeadParameter::new(device.clone(), context_dim, vocab_size, &lm_weight)?;
        let dense_sparse_buffer = GpuBuffer::from_f32(&device, &dense_sparse_grad)?;
        let mut preaccumulate = vulkan::ComputeBatch::new(&device)?;
        reference_shared.record_zero_grad(&mut preaccumulate)?;
        reference_shared.record_accumulate_gradient(&mut preaccumulate, &dense_sparse_buffer)?;
        preaccumulate.submit()?;
        let mut reference = HierarchosOutNormHeadTrainer::from_shared_lm_head(
            reference_shared.clone(),
            token_count,
            &norm_weight,
            &norm_bias,
        )?;
        let reference_step = reference.train_step_finalize_shared_lm(&hidden, &targets, hyper)?;

        assert_eq!(radix_step.step, 1);
        assert_eq!(reference_step.step, 1);
        assert_eq!(radix_shared.step(), reference_shared.step());
        assert_eq!(radix_step.loss, reference_step.loss);
        assert_eq!(radix_shared.weights()?, reference_shared.weights()?);
        Ok(())
    }

    #[test]
    fn dense_lm_shared_gradient_accumulates_across_submissions() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let initial = [0.10, -0.20, 0.30, 0.40, -0.50, 0.60];
        let norm_weight = [1.0, 0.75];
        let norm_bias = [0.05, -0.10];
        let hidden_a = [0.25, -0.75, 0.60, 0.20];
        let hidden_b = [-0.40, 0.35, 0.15, -0.55];
        let targets_a = [1, 2];
        let targets_b = [2, 0];

        let direct_shared = SharedLmHeadParameter::new(device.clone(), 2, 3, &initial)?;
        let direct = HierarchosOutNormHeadTrainer::from_shared_lm_head(
            direct_shared.clone(),
            2,
            &norm_weight,
            &norm_bias,
        )?;
        let staged_shared = SharedLmHeadParameter::new(device.clone(), 2, 3, &initial)?;
        let staged = HierarchosOutNormHeadTrainer::from_shared_lm_head(
            staged_shared.clone(),
            2,
            &norm_weight,
            &norm_bias,
        )?;

        let direct_input_a = GpuBuffer::from_f32(&device, &hidden_a)?;
        let mut direct_first = vulkan::ComputeBatch::new(&device)?;
        direct_shared.record_zero_grad(&mut direct_first)?;
        direct_first.upload_u32(&direct.targets, &targets_a)?;
        let direct_push_a =
            direct.record_norm_head_stats(&mut direct_first, &direct_input_a, targets_a.len())?;
        direct.record_streaming_lm_loss_backward_with_dense_target(
            &mut direct_first,
            &direct_input_a,
            targets_a.len(),
            direct_push_a,
            LmWeightGradWriteMode::SharedOverwrite,
            0.0,
            false,
        )?;
        direct_first.submit()?;

        let direct_input_b = GpuBuffer::from_f32(&device, &hidden_b)?;
        let mut direct_second = vulkan::ComputeBatch::new(&device)?;
        direct_second.upload_u32(&direct.targets, &targets_b)?;
        let direct_push_b =
            direct.record_norm_head_stats(&mut direct_second, &direct_input_b, targets_b.len())?;
        direct.record_streaming_lm_loss_backward_with_dense_target(
            &mut direct_second,
            &direct_input_b,
            targets_b.len(),
            direct_push_b,
            LmWeightGradWriteMode::SharedAccumulate,
            0.0,
            false,
        )?;
        direct_second.submit()?;

        let staged_input_a = GpuBuffer::from_f32(&device, &hidden_a)?;
        let mut staged_first = vulkan::ComputeBatch::new(&device)?;
        staged_shared.record_zero_grad(&mut staged_first)?;
        staged_first.upload_u32(&staged.targets, &targets_a)?;
        let staged_push_a =
            staged.record_norm_head_stats(&mut staged_first, &staged_input_a, targets_a.len())?;
        staged.record_streaming_lm_loss_backward_with_dense_target(
            &mut staged_first,
            &staged_input_a,
            targets_a.len(),
            staged_push_a,
            LmWeightGradWriteMode::ScratchOverwrite,
            0.0,
            false,
        )?;
        staged_first.submit()?;

        let staged_input_b = GpuBuffer::from_f32(&device, &hidden_b)?;
        let mut staged_second = vulkan::ComputeBatch::new(&device)?;
        staged_second.upload_u32(&staged.targets, &targets_b)?;
        let staged_push_b =
            staged.record_norm_head_stats(&mut staged_second, &staged_input_b, targets_b.len())?;
        staged.record_streaming_lm_loss_backward_with_dense_target(
            &mut staged_second,
            &staged_input_b,
            targets_b.len(),
            staged_push_b,
            LmWeightGradWriteMode::ScratchOverwrite,
            0.0,
            false,
        )?;
        staged_second.submit()?;

        let direct_grad = direct_shared.gradient_buffer().read_f32(initial.len())?;
        let staged_grad = staged_shared.gradient_buffer().read_f32(initial.len())?;
        let max_abs = direct_grad
            .iter()
            .zip(&staged_grad)
            .map(|(direct, staged)| (direct - staged).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 2.0e-6,
            "direct cross-submit LM dW accumulation drifted from staged reference by {max_abs}"
        );
        Ok(())
    }
}
