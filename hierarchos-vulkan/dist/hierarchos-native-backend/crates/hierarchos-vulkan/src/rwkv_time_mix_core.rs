use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::rwkv_low_rank::{
    RwkvLowRankFanInSchedule, RwkvLowRankFp16ParameterMirrors, RwkvLowRankParameterGradArithmetic,
};
use crate::rwkv_optimizer::{RwkvDecayClass, RwkvTrainableRef};
use crate::{
    read_f32_tensor, vulkan, GpuBuffer, RwkvLowRankOp, RwkvLowRankResult, RwkvPostMixOp,
    RwkvPostMixResult, VulkanDevice,
};

const TIME_MIX_FORWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_time_mix3_forward.spv");
const TIME_MIX_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_time_mix3_backward.spv");
const LINEAR3_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear3_forward.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_fused.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_FUSED_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_fused_wg32.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_FUSED_TWO_ROWS_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_fused_two_rows.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE2_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE2_TILE32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2_tile32.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE2_TILE64_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2_tile64.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE4_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse4.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE4_TILE32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse4_tile32.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE4_TILE64_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse4_tile64.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE8_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse8.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE8_TILE32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse8_tile32.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE8_TILE64_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse8_tile64.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_wg32.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_TAPE_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_tape.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_TAPE_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_tape_wg32.spv");
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE2_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE2_TILE32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2_tile32.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE2_TILE64_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2_tile64.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE4_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse4.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE4_TILE32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse4_tile32.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE4_TILE64_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse4_tile64.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE8_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse8.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE8_TILE32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse8_tile32.spv"
);
const TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE8_TILE64_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse8_tile64.spv"
);
const TIME_MIX_WEIGHT_REUSE2_TILE16_SHARED_BYTES: u32 = 15_232;
const TIME_MIX_WEIGHT_REUSE2_TILE32_SHARED_BYTES: u32 = 27_904;
const TIME_MIX_WEIGHT_REUSE2_TILE64_SHARED_BYTES: u32 = 53_248;
const TIME_MIX_WEIGHT_REUSE4_TILE16_SHARED_BYTES: u32 = 18_176;
const TIME_MIX_WEIGHT_REUSE4_TILE32_SHARED_BYTES: u32 = 31_232;
const TIME_MIX_WEIGHT_REUSE4_TILE64_SHARED_BYTES: u32 = 57_344;
const TIME_MIX_WEIGHT_REUSE8_TILE16_SHARED_BYTES: u32 = 24_064;
const TIME_MIX_WEIGHT_REUSE8_TILE32_SHARED_BYTES: u32 = 37_888;
const TIME_MIX_WEIGHT_REUSE8_TILE64_SHARED_BYTES: u32 = 65_536;
const HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE_ENV: &str =
    "HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE";
const HIERARCHOS_RWKV_TIME_MIX_ENABLE_TWO_ROW_FORWARD_FUSION_ENV: &str =
    "HIERARCHOS_RWKV_TIME_MIX_ENABLE_TWO_ROW_FORWARD_FUSION";
const HIERARCHOS_RWKV_TIME_MIX_ENABLE_WEIGHT_REUSE2_ENV: &str =
    "HIERARCHOS_RWKV_TIME_MIX_ENABLE_WEIGHT_REUSE2";
const HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_DISABLE_AUTOTUNE_ENV: &str =
    "HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_DISABLE_AUTOTUNE";
const HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_AUTOTUNE_LOG_ENV: &str =
    "HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_AUTOTUNE_LOG";
const HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_DISABLE_AUTOTUNE_ENV: &str =
    "HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_DISABLE_AUTOTUNE";
const HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_AUTOTUNE_LOG_ENV: &str =
    "HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_AUTOTUNE_LOG";
const HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_DISABLE_AUTOTUNE_ENV: &str =
    "HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_DISABLE_AUTOTUNE";
const HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_AUTOTUNE_LOG_ENV: &str =
    "HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_AUTOTUNE_LOG";
const LINEAR3_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear3_weight_grad.spv");
const LINEAR3_WEIGHT_GRAD_TILED_SPV: &[u8] =
    include_bytes!("../shaders/linear3_weight_grad_tiled.spv");
const LINEAR3_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear3_input_grad.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse2.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE32_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse2_tile32.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE64_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse2_tile64.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE4_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse4.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE4_TILE32_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse4_tile32.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE4_TILE64_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse4_tile64.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE8_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse8.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE8_TILE32_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse8_tile32.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE8_TILE64_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_weight_reuse8_tile64.spv");
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE16_SHARED_BYTES: u32 = 12_288;
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE32_SHARED_BYTES: u32 = 24_576;
const LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE64_SHARED_BYTES: u32 = 49_152;
const LINEAR3_INPUT_GRAD_FP16_SCALED_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_fp16_scaled.spv");
const LINEAR3_INPUT_GRAD_FP16_SOURCE_SCALED_SPV: &[u8] =
    include_bytes!("../shaders/linear3_input_grad_fp16_source_scaled.spv");
const LINEAR3_TIME_MIX_BACKWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/linear3_time_mix_backward_fused.spv");
const KEY_TRANSFORM_FORWARD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_key_transform_forward.spv");
const KEY_TRANSFORM_BACKWARD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_key_transform_backward.spv");
const KEY_TRANSFORM_BACKWARD_SUBGROUP_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_key_transform_backward_subgroup.spv");
const KEY_TRANSFORM_PARAM_REDUCE_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_key_transform_param_reduce.spv");
const RWKV_MATRIX_STATE_FORWARD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_forward.spv");
const RWKV_KEY_STATE_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_key_state_forward_fused.spv");
const RWKV_MATRIX_STATE_BACKWARD_ROWS_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_rows.spv");
const RWKV_MATRIX_STATE_BACKWARD_COLS_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_cols.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RK_ADD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rk_add.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_wg32.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_WG128_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_wg128.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_RECURRENT_SUBGROUP_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_recurrent_subgroup.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_RECURRENT_SUBGROUP_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_recurrent_subgroup_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_RECURRENT_SUBGROUP_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_recurrent_subgroup_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_packed.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_wg32.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_WG128_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_wg128.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SUBGROUP_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_subgroup.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SUBGROUP_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_subgroup_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SUBGROUP_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_subgroup_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_RECURRENT_SUBGROUP_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_recurrent_subgroup.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_RECURRENT_SUBGROUP_WG32_SPV: &[u8] =
    include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_RECURRENT_SUBGROUP_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TREE_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_tree.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TREE_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_tree_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TREE_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_tree_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TILED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_tiled.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TILED_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_tiled_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TILED_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_tiled_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce.spv");
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SUBGROUP_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_subgroup.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SUBGROUP_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SUBGROUP_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_RECURRENT_SUBGROUP_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_RECURRENT_SUBGROUP_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_RECURRENT_SUBGROUP_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TREE_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_tree.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TREE_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_tree_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TREE_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_tree_wg128.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TILED_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_tiled.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TILED_WG32_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg32.spv"
);
const RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TILED_WG128_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg128.spv"
);
const VECTOR_ADD_SPV: &[u8] = include_bytes!("../shaders/vector_add.spv");
const VECTOR_ADD3_SPV: &[u8] = include_bytes!("../shaders/vector_add3.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MixPush {
    batch: u32,
    width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct KeyPush {
    batch: u32,
    width: u32,
    head_size: u32,
}

/// Numerical contract for RWKV backward reductions.
///
/// `StrictParity` preserves the historical serial FP32 accumulation order used
/// by the PyTorch/SafeTensors parity suite. `FastSubgroup` is an explicit
/// experimental arm that permits subgroup arithmetic for key-normalization dot
/// products. `FastRecurrentTree` instead keeps key normalization strict and
/// parallelizes recurrent row/column dot products with interleaved lane
/// partials plus shared-memory trees. `FastRecurrentTiled` uses contiguous lane
/// tiles before the shared-memory merge so devices can trade memory access
/// locality against accumulation order. `FastRecurrentSubgroup` uses exact
/// hardware-subgroup geometry: a one-subgroup workgroup gives one recurrent row
/// to each lane and preserves the historical serial column-dot order. Wider
/// workgroups made from complete subgroups either assign independent rows and
/// columns to those waves when one subgroup spans a head, or pair two subgroups
/// per head-local reduction when (notably on NVIDIA wave32 hardware) a 64-wide
/// head cannot fit in one wave. The paired-wave arm crosses subgroup boundaries
/// only through small shared-memory partials plus the existing `q(row)` handoff.
/// In deeply fused variants key normalization follows the same two-wave contract
/// without changing the tensor/checkpoint contract.
/// Fast arms may differ by a few FP32 ulps while retaining the same tensors,
/// optimizer state, and checkpoint format.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum RwkvNumericsPolicy {
    #[default]
    StrictParity,
    FastSubgroup,
    FastRecurrentTree,
    FastRecurrentTiled,
    FastRecurrentSubgroup,
}

impl RwkvNumericsPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::StrictParity => "strict-parity",
            Self::FastSubgroup => "fast-subgroup",
            Self::FastRecurrentTree => "fast-recurrent-tree",
            Self::FastRecurrentTiled => "fast-recurrent-tiled",
            Self::FastRecurrentSubgroup => "fast-recurrent-subgroup",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "strict" | "strict-parity" => Some(Self::StrictParity),
            "fast-subgroup" => Some(Self::FastSubgroup),
            "fast-recurrent-tree" | "recurrent-tree" | "tree" => Some(Self::FastRecurrentTree),
            "fast-recurrent-tiled" | "recurrent-tiled" | "tiled" => Some(Self::FastRecurrentTiled),
            "fast-recurrent-subgroup" | "recurrent-subgroup" | "subgroup-recurrent" => {
                Some(Self::FastRecurrentSubgroup)
            }
            _ => None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StatePush {
    batch: u32,
    width: u32,
    head_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackedStatePush {
    batch: u32,
    width: u32,
    head_size: u32,
    matrix_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackedFastStatePush {
    batch: u32,
    width: u32,
    head_size: u32,
    matrix_offset: u32,
    state_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
enum StateBackwardSchedule {
    RkvAdd3,
    RkvAdd3KeyTransform,
    RkvAdd3KeyTransformReduce,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum BackwardKernelGeometry {
    Wg32,
    Wg64,
    Wg128,
}

impl BackwardKernelGeometry {
    const ALL: [Self; 3] = [Self::Wg32, Self::Wg64, Self::Wg128];

    fn label(self) -> &'static str {
        match self {
            Self::Wg32 => "rwkv-state-bwd-wg32",
            Self::Wg64 => "rwkv-state-bwd-wg64",
            Self::Wg128 => "rwkv-state-bwd-wg128",
        }
    }

    fn workgroup_size(self) -> u32 {
        match self {
            Self::Wg32 => 32,
            Self::Wg64 => 64,
            Self::Wg128 => 128,
        }
    }
}

fn backward_kernel_geometry_priority(
    geometry: BackwardKernelGeometry,
    subgroup_size: u32,
) -> (u8, u32) {
    let subgroup_size = subgroup_size.max(1);
    let workgroup_size = geometry.workgroup_size();
    let alignment_class = if workgroup_size == subgroup_size {
        0
    } else if workgroup_size.is_multiple_of(subgroup_size) {
        1
    } else if subgroup_size.is_multiple_of(workgroup_size) {
        2
    } else {
        3
    };
    (alignment_class, workgroup_size.abs_diff(subgroup_size))
}

fn recurrent_subgroup_geometry_supported(
    workgroup_size: u32,
    subgroup_size: u32,
    head_size: usize,
) -> bool {
    if subgroup_size == 0
        || workgroup_size < subgroup_size
        || !workgroup_size.is_multiple_of(subgroup_size)
        || head_size == 0
    {
        return false;
    }

    let subgroups_per_workgroup = workgroup_size / subgroup_size;
    let subgroups_per_head = head_size.div_ceil(subgroup_size as usize) as u32;
    subgroups_per_head <= 2 && subgroups_per_workgroup >= subgroups_per_head
}

impl StateBackwardSchedule {
    fn label(self) -> &'static str {
        match self {
            Self::RkvAdd3 => "rkv-add3+key+reduce",
            Self::RkvAdd3KeyTransform => "rkv-add3-key+reduce",
            Self::RkvAdd3KeyTransformReduce => "rkv-add3-key-reduce",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StateBackwardAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    width: usize,
    head_size: usize,
    batch: usize,
}

#[derive(Clone, Copy, Debug)]
struct StateBackwardDecision {
    schedule: StateBackwardSchedule,
    autotuned: bool,
}

static STATE_BACKWARD_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<StateBackwardAutotuneKey, StateBackwardDecision>>,
> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ForwardProjectionRecurrence {
    Full,
    Packed,
}

impl ForwardProjectionRecurrence {
    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Packed => "packed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ForwardProjectionTopology {
    Baseline,
    WeightReuse { rows: u32, tile: u32 },
}

impl ForwardProjectionTopology {
    fn label(self) -> String {
        match self {
            Self::Baseline => "baseline".to_owned(),
            Self::WeightReuse { rows, tile } => format!("reuse{rows}-tile{tile}"),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ForwardProjectionAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    width: usize,
    head_size: usize,
    batch_pairs: usize,
    has_unpaired_tail: bool,
    recurrence: ForwardProjectionRecurrence,
}

#[derive(Clone, Copy, Debug)]
struct ForwardProjectionDecision {
    topology: ForwardProjectionTopology,
    autotuned: bool,
}

static FORWARD_PROJECTION_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<ForwardProjectionAutotuneKey, ForwardProjectionDecision>>,
> = OnceLock::new();

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ProjectionInputGradAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    width: usize,
    batch_pairs: usize,
    has_unpaired_tail: bool,
}

#[derive(Clone, Copy, Debug)]
struct ProjectionInputGradDecision {
    topology: ProjectionInputGradTopology,
    autotuned: bool,
}

static PROJECTION_INPUT_GRAD_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<ProjectionInputGradAutotuneKey, ProjectionInputGradDecision>>,
> = OnceLock::new();

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ProjectionWeightGradAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    width: usize,
    rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProjectionWeightGradTopology {
    Baseline,
    Tiled,
}

impl ProjectionWeightGradTopology {
    fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Tiled => "tiled16x16",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ProjectionWeightGradDecision {
    topology: ProjectionWeightGradTopology,
    autotuned: bool,
}

static PROJECTION_WEIGHT_GRAD_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<ProjectionWeightGradAutotuneKey, ProjectionWeightGradDecision>>,
> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProjectionInputGradTopology {
    Baseline,
    WeightReuse { rows: u32, tile: u32 },
}

impl ProjectionInputGradTopology {
    fn label(self) -> String {
        match self {
            Self::Baseline => "baseline".to_owned(),
            Self::WeightReuse { rows, tile } => format!("reuse{rows}-tile{tile}"),
        }
    }
}

#[derive(Default)]
struct WeightReuseKernels {
    tile16: Option<vulkan::ComputeKernel>,
    tile32: Option<vulkan::ComputeKernel>,
    tile64: Option<vulkan::ComputeKernel>,
}

impl WeightReuseKernels {
    fn get(&self, tile: u32) -> Option<&vulkan::ComputeKernel> {
        match tile {
            16 => self.tile16.as_ref(),
            32 => self.tile32.as_ref(),
            64 => self.tile64.as_ref(),
            _ => None,
        }
    }

    fn available_tiles(&self) -> Vec<u32> {
        let mut tiles = Vec::with_capacity(3);
        if self.tile16.is_some() {
            tiles.push(16);
        }
        if self.tile32.is_some() {
            tiles.push(32);
        }
        if self.tile64.is_some() {
            tiles.push(64);
        }
        tiles
    }

    fn is_empty(&self) -> bool {
        self.tile16.is_none() && self.tile32.is_none() && self.tile64.is_none()
    }

    fn largest_tile(&self) -> Option<u32> {
        if self.tile64.is_some() {
            Some(64)
        } else if self.tile32.is_some() {
            Some(32)
        } else if self.tile16.is_some() {
            Some(16)
        } else {
            None
        }
    }
}

#[derive(Default)]
struct MultiRowWeightReuseKernels {
    rows2: WeightReuseKernels,
    rows4: WeightReuseKernels,
    rows8: WeightReuseKernels,
}

impl MultiRowWeightReuseKernels {
    fn get(&self, rows: u32, tile: u32) -> Option<&vulkan::ComputeKernel> {
        match rows {
            2 => self.rows2.get(tile),
            4 => self.rows4.get(tile),
            8 => self.rows8.get(tile),
            _ => None,
        }
    }

    fn rows(&self, rows: u32) -> Option<&WeightReuseKernels> {
        match rows {
            2 => Some(&self.rows2),
            4 => Some(&self.rows4),
            8 => Some(&self.rows8),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
enum ProjectionBackwardSchedule {
    WeightThenFusedMix,
    FusedMixThenWeight,
    WeightThenSplitMix,
    SplitMixThenWeight,
}

impl ProjectionBackwardSchedule {
    fn label(self) -> &'static str {
        match self {
            Self::WeightThenFusedMix => "weight->fused-input-mix",
            Self::FusedMixThenWeight => "fused-input-mix->weight",
            Self::WeightThenSplitMix => "weight->split-input-mix",
            Self::SplitMixThenWeight => "split-input-mix->weight",
        }
    }

    fn uses_fused_mix(self) -> bool {
        matches!(self, Self::WeightThenFusedMix | Self::FusedMixThenWeight)
    }

    fn weight_first(self) -> bool {
        matches!(self, Self::WeightThenFusedMix | Self::WeightThenSplitMix)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct BackwardSegmentSchedule {
    state: StateBackwardSchedule,
    projection: ProjectionBackwardSchedule,
    low_rank_fan_in: Option<RwkvLowRankFanInSchedule>,
}

impl BackwardSegmentSchedule {
    fn label(self) -> String {
        let base = format!("{}+{}", self.state.label(), self.projection.label());
        match self.low_rank_fan_in {
            Some(fan_in) => format!("{base}+{}", fan_in.label()),
            None => base,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct BackwardSegmentAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    width: usize,
    head_size: usize,
    batch: usize,
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    full_cell: bool,
}

#[derive(Clone, Copy, Debug)]
struct BackwardSegmentDecision {
    schedule: BackwardSegmentSchedule,
    autotuned: bool,
}

static BACKWARD_SEGMENT_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<BackwardSegmentAutotuneKey, BackwardSegmentDecision>>,
> = OnceLock::new();

const BACKWARD_SEGMENT_PERSISTENT_CACHE_VERSION: u32 = 1;
const BACKWARD_SEGMENT_ELIMINATION_THRESHOLD: usize = 8;

#[derive(Debug, Deserialize, Serialize)]
struct PersistentBackwardSegmentEntry {
    key: BackwardSegmentAutotuneKey,
    schedule: BackwardSegmentSchedule,
}

#[derive(Debug, Deserialize, Serialize)]
struct PersistentBackwardSegmentCache {
    version: u32,
    entries: Vec<PersistentBackwardSegmentEntry>,
}

impl Default for PersistentBackwardSegmentCache {
    fn default() -> Self {
        Self {
            version: BACKWARD_SEGMENT_PERSISTENT_CACHE_VERSION,
            entries: Vec::new(),
        }
    }
}

static BACKWARD_SEGMENT_PERSISTENT_CACHE_IO: OnceLock<Mutex<()>> = OnceLock::new();

struct LayerNormForwardInput<'a> {
    x: &'a GpuBuffer,
    weight: &'a GpuBuffer,
    bias: &'a GpuBuffer,
    mean: &'a GpuBuffer,
    rstd: &'a GpuBuffer,
    eps: f32,
}

#[derive(Debug)]
pub struct RwkvTimeMixCoreResult {
    pub new_state: Vec<f32>,
    pub tmix: Vec<f32>,
    pub scaled_k: Vec<f32>,
    pub kk: Vec<f32>,
    pub grad_state: Vec<f32>,
    pub grad_x_norm: Vec<f32>,
    pub grad_previous: Vec<f32>,
    pub grad_a: Vec<f32>,
    pub grad_w: Vec<f32>,
    pub grad_mix_r: Vec<f32>,
    pub grad_mix_k: Vec<f32>,
    pub grad_mix_v: Vec<f32>,
    pub grad_receptance_weight: Vec<f32>,
    pub grad_key_weight: Vec<f32>,
    pub grad_value_weight: Vec<f32>,
    pub grad_k_k: Vec<f32>,
    pub grad_k_a: Vec<f32>,
}

#[derive(Debug)]
pub struct RwkvFusedTimeMixCoreResult {
    pub core: RwkvTimeMixCoreResult,
    pub low_rank: RwkvLowRankResult,
    /// Sum of the r/k/v and a/w/g branches at the shared normalized input.
    pub grad_x_norm: Vec<f32>,
    /// Sum of the r/k/v and a/w/g branches at the shared previous-token edge.
    pub grad_previous: Vec<f32>,
}

#[derive(Debug)]
pub struct RwkvFullTimeMixResult {
    pub core: RwkvTimeMixCoreResult,
    pub low_rank: RwkvLowRankResult,
    pub post_mix: RwkvPostMixResult,
    /// Complete time-mix gradient at the shared normalized input, including
    /// r/k/v, a/w/g, GroupNorm, bonus, gate, and output-projection consumers.
    pub grad_x_norm: Vec<f32>,
    pub grad_previous: Vec<f32>,
}

/// Composed Vulkan-native RWKV-v8 recurrent training slice.
///
/// This operation owns the complete r/k/v side of the PyTorch RWKV time-mix
/// graph around the matrix recurrence:
///
/// `x_norm + previous -> xr/xk/xv -> r/raw_k/v projections`
/// `raw_k -> normalized kk + in-context scaled k`
/// `state + r/k/v/kk/a/w -> new_state + tmix`
///
/// Its backward pass continues through key normalization/scaling, all three
/// projection matrices, and the three time-mix coefficients. Consequently the
/// caller only supplies gradients at the two true consumers of the recurrent
/// primitive (`new_state` and `tmix`), rather than Python-computed gradients for
/// projected r/k/v tensors.
///
/// Projection weights use the exact PyTorch `nn.Linear` row-major
/// `[out_features, in_features]` layout, so weights can be copied directly from
/// the same safetensors used by CUDA/PyTorch inference.
pub struct RwkvTimeMixCoreOp {
    device: VulkanDevice,
    width: usize,
    head_size: usize,
    heads: usize,
    max_batch: usize,
    low_rank: Option<RwkvLowRankOp>,
    post_mix: Option<RwkvPostMixOp>,

    mix_r: GpuBuffer,
    mix_k: GpuBuffer,
    mix_v: GpuBuffer,
    receptance_weight: GpuBuffer,
    key_weight: GpuBuffer,
    value_weight: GpuBuffer,
    k_k: GpuBuffer,
    k_a: GpuBuffer,

    state: GpuBuffer,
    x_norm: GpuBuffer,
    previous: GpuBuffer,
    a: GpuBuffer,
    w: GpuBuffer,
    grad_new_state: GpuBuffer,
    grad_tmix: GpuBuffer,

    xr: GpuBuffer,
    xk: GpuBuffer,
    xv: GpuBuffer,
    r: GpuBuffer,
    raw_k: GpuBuffer,
    v: GpuBuffer,
    scaled_k: GpuBuffer,
    kk: GpuBuffer,
    new_state: GpuBuffer,
    tmix: GpuBuffer,
    saved_sa: GpuBuffer,
    saved_q: GpuBuffer,

    grad_state: GpuBuffer,
    grad_r: GpuBuffer,
    grad_scaled_k: GpuBuffer,
    grad_v: GpuBuffer,
    grad_kk: GpuBuffer,
    grad_a_direct: GpuBuffer,
    grad_w: GpuBuffer,
    grad_raw_k: GpuBuffer,
    grad_a: GpuBuffer,
    grad_k_k_partial: GpuBuffer,
    grad_k_a_partial: GpuBuffer,
    grad_k_k: GpuBuffer,
    grad_k_a: GpuBuffer,
    grad_xr: GpuBuffer,
    grad_xk: GpuBuffer,
    grad_xv: GpuBuffer,
    grad_receptance_weight: GpuBuffer,
    grad_key_weight: GpuBuffer,
    grad_value_weight: GpuBuffer,
    grad_x_norm: GpuBuffer,
    grad_previous: GpuBuffer,
    grad_mix_r: GpuBuffer,
    grad_mix_k: GpuBuffer,
    grad_mix_v: GpuBuffer,
    total_grad_r: GpuBuffer,
    total_grad_scaled_k: GpuBuffer,
    total_grad_v: GpuBuffer,
    total_grad_v_external: GpuBuffer,
    fused_grad_x_norm: GpuBuffer,
    fused_grad_previous: GpuBuffer,
    grad_g_external: GpuBuffer,

    new_state_readback: GpuBuffer,
    tmix_readback: GpuBuffer,
    scaled_k_readback: GpuBuffer,
    kk_readback: GpuBuffer,
    grad_state_readback: GpuBuffer,
    grad_x_norm_readback: GpuBuffer,
    grad_previous_readback: GpuBuffer,
    grad_a_readback: GpuBuffer,
    grad_w_readback: GpuBuffer,
    grad_mix_r_readback: GpuBuffer,
    grad_mix_k_readback: GpuBuffer,
    grad_mix_v_readback: GpuBuffer,
    grad_receptance_weight_readback: GpuBuffer,
    grad_key_weight_readback: GpuBuffer,
    grad_value_weight_readback: GpuBuffer,
    grad_k_k_readback: GpuBuffer,
    grad_k_a_readback: GpuBuffer,
    fused_grad_x_norm_readback: GpuBuffer,
    fused_grad_previous_readback: GpuBuffer,

    time_mix_forward: vulkan::ComputeKernel,
    time_mix_backward: vulkan::ComputeKernel,
    linear3_forward: vulkan::ComputeKernel,
    time_mix_linear3_key_state_forward_fused: Option<vulkan::ComputeKernel>,
    time_mix_linear3_key_state_forward_fused_two_rows: Option<vulkan::ComputeKernel>,
    time_mix_linear3_key_state_forward_weight_reuse: MultiRowWeightReuseKernels,
    time_mix_linear3_key_state_forward_packed_fast: Option<vulkan::ComputeKernel>,
    time_mix_linear3_key_state_forward_packed_fast_weight_reuse: MultiRowWeightReuseKernels,
    time_mix_linear3_key_state_forward_packed_tape: Option<vulkan::ComputeKernel>,
    linear3_weight_grad: vulkan::ComputeKernel,
    linear3_weight_grad_tiled: Option<vulkan::ComputeKernel>,
    linear3_input_grad: vulkan::ComputeKernel,
    linear3_input_grad_weight_reuse: MultiRowWeightReuseKernels,
    linear3_input_grad_fp16_scaled: Option<vulkan::ComputeKernel>,
    linear3_input_grad_fp16_source_scaled: Option<vulkan::ComputeKernel>,
    native_fp16_projection_input_grad: bool,
    backward_source_scale: f32,
    source_scaled_backward_domain: bool,
    linear3_time_mix_backward_fused: Option<vulkan::ComputeKernel>,
    key_transform_forward: vulkan::ComputeKernel,
    key_transform_backward: vulkan::ComputeKernel,
    key_transform_backward_subgroup: Option<vulkan::ComputeKernel>,
    key_transform_param_reduce: vulkan::ComputeKernel,
    state_forward: vulkan::ComputeKernel,
    key_state_forward_fused: vulkan::ComputeKernel,
    state_backward_rows: vulkan::ComputeKernel,
    state_backward_cols: vulkan::ComputeKernel,
    state_backward_fused: vulkan::ComputeKernel,
    state_backward_fused_rk_add: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_recurrent_subgroup: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_recurrent_subgroup_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_recurrent_subgroup_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_packed: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_subgroup: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_subgroup_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_subgroup_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_recurrent_subgroup: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg32:
        Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg128:
        Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_tree: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_tree_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_tree_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_tiled: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_tiled_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_tiled_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_subgroup: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg128:
        Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup:
        Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg32:
        Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg128:
        Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_tree: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_tree_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_tree_wg128: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_tiled: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg32: Option<vulkan::ComputeKernel>,
    state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg128: Option<vulkan::ComputeKernel>,
    numerics_policy: RwkvNumericsPolicy,
    state_backward_schedule_batch1: Option<StateBackwardDecision>,
    state_backward_schedule_multi: Option<StateBackwardDecision>,
    state_backward_profile_batch: usize,
    backward_segment_schedule_batch1: Option<BackwardSegmentDecision>,
    backward_segment_schedule_multi: Option<BackwardSegmentDecision>,
    backward_kernel_geometry_batch1: BackwardKernelGeometry,
    backward_kernel_geometry_multi: BackwardKernelGeometry,
    vector_add: vulkan::ComputeKernel,
    vector_add3: vulkan::ComputeKernel,
}

fn create_state_backward_geometry_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
    binding_count: usize,
    workgroup_size: u32,
) -> Result<Option<vulkan::ComputeKernel>> {
    if !device.supports_storage_buffer_bindings(binding_count as u32)
        || !device.supports_compute_work_group_size_x(workgroup_size)
    {
        return Ok(None);
    }

    let mut accesses = vec![vulkan::BindingAccess::ReadOnly; binding_count];
    accesses[15..22].fill(vulkan::BindingAccess::MayWrite);
    if binding_count >= 29 {
        accesses[25..29].fill(vulkan::BindingAccess::MayWrite);
    }
    if binding_count >= 31 {
        accesses[29..31].fill(vulkan::BindingAccess::MayWrite);
    }
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &accesses,
        std::mem::size_of::<StatePush>() as u32,
    )?))
}

fn create_state_backward_packed_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
) -> Result<Option<vulkan::ComputeKernel>> {
    const BINDING_COUNT: usize = 29;
    if !device.supports_storage_buffer_bindings(BINDING_COUNT as u32)
        || !device.supports_compute_work_group_size_x(64)
    {
        return Ok(None);
    }

    let mut accesses = vec![vulkan::BindingAccess::ReadOnly; BINDING_COUNT];
    accesses[15..22].fill(vulkan::BindingAccess::MayWrite);
    accesses[25..29].fill(vulkan::BindingAccess::MayWrite);
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &accesses,
        std::mem::size_of::<PackedStatePush>() as u32,
    )?))
}

fn create_time_mix_forward_fused_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
) -> Result<vulkan::ComputeKernel> {
    vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
        ],
        std::mem::size_of::<StatePush>() as u32,
    )
}

fn create_time_mix_forward_packed_fast_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
) -> Result<vulkan::ComputeKernel> {
    vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::ReadOnly,
        ],
        std::mem::size_of::<PackedFastStatePush>() as u32,
    )
}

fn create_time_mix_forward_packed_tape_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
) -> Result<vulkan::ComputeKernel> {
    let mut accesses = vec![vulkan::BindingAccess::ReadOnly; 24];
    accesses[12..23].fill(vulkan::BindingAccess::MayWrite);
    vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &accesses,
        std::mem::size_of::<PackedStatePush>() as u32,
    )
}

impl RwkvTimeMixCoreOp {
    /// Load the r/k/v recurrent-core parameters directly from Hierarchos'
    /// standard `model.safetensors`. `prefix` is normally `h_rnn` or `l_rnn`.
    /// No tensor is transposed or renamed: the GPU buffers preserve the same
    /// PyTorch row-major layouts used by CUDA inference.
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        head_size: usize,
        max_batch: usize,
    ) -> Result<Self> {
        if prefix.trim().is_empty() {
            bail!("RWKV tensor prefix must not be empty");
        }
        let tensor_path = model_dir.as_ref().join("model.safetensors");

        let (mix_r_shape, mix_r) = read_f32_tensor(&tensor_path, &format!("{prefix}.x_r"))?;
        let width = vector_width(&mix_r_shape).with_context(|| {
            format!("RWKV tensor {prefix}.x_r must have shape [C] or [1, C], got {mix_r_shape:?}")
        })?;

        let mix_k = read_vector_tensor(&tensor_path, &format!("{prefix}.x_k"), width)?;
        let mix_v = read_vector_tensor(&tensor_path, &format!("{prefix}.x_v"), width)?;
        let k_k = read_vector_tensor(&tensor_path, &format!("{prefix}.k_k"), width)?;
        let k_a = read_vector_tensor(&tensor_path, &format!("{prefix}.k_a"), width)?;
        let receptance_weight =
            read_matrix_tensor(&tensor_path, &format!("{prefix}.receptance.weight"), width)?;
        let key_weight = read_matrix_tensor(&tensor_path, &format!("{prefix}.key.weight"), width)?;
        let value_weight =
            read_matrix_tensor(&tensor_path, &format!("{prefix}.value.weight"), width)?;

        Self::new(
            device,
            width,
            head_size,
            max_batch,
            &mix_r,
            &mix_k,
            &mix_v,
            &receptance_weight,
            &key_weight,
            &value_weight,
            &k_k,
            &k_a,
        )
    }

    /// Load the recurrent r/k/v core and the low-rank a/w/g branches from the
    /// same standard Hierarchos SafeTensors package. The resulting op exposes a
    /// single-submit fused training path in which a/w are never supplied by the
    /// host and g remains resident on the Vulkan device for downstream cell
    /// composition.
    pub fn from_model_package_fused(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        head_size: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut op =
            Self::from_model_package(device.clone(), model_dir, prefix, head_size, max_batch)?;
        let low_rank = RwkvLowRankOp::from_model_package(device, model_dir, prefix, max_batch)?;
        if low_rank.width() != op.width {
            bail!(
                "RWKV fused low-rank width {} does not match recurrent-core width {}",
                low_rank.width(),
                op.width
            );
        }
        if low_rank.max_batch() < op.max_batch {
            bail!(
                "RWKV fused low-rank capacity {} is smaller than recurrent-core capacity {}",
                low_rank.max_batch(),
                op.max_batch
            );
        }
        op.low_rank = Some(low_rank);
        Ok(op)
    }

    /// Load the full RWKV time-mix trainable slice through the output
    /// projection. This extends the single-submit a/w/g + recurrent graph with
    /// GroupNorm, the receptance/key `r_k` bonus, g gating, and `output.weight`.
    pub fn from_model_package_full(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        head_size: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut op = Self::from_model_package_fused(
            device.clone(),
            model_dir,
            prefix,
            head_size,
            max_batch,
        )?;
        let post_mix =
            RwkvPostMixOp::from_model_package(device, model_dir, prefix, head_size, max_batch)?;
        if post_mix.width() != op.width || post_mix.head_size() != op.head_size {
            bail!(
                "RWKV post-mix geometry width/head_size={}/{} does not match recurrent core {}/{}",
                post_mix.width(),
                post_mix.head_size(),
                op.width,
                op.head_size
            );
        }
        if post_mix.max_batch() < op.max_batch {
            bail!(
                "RWKV post-mix capacity {} is smaller than recurrent-core capacity {}",
                post_mix.max_batch(),
                op.max_batch
            );
        }
        op.post_mix = Some(post_mix);
        // Re-run the recurrence/projection schedule search with the complete
        // post-mix -> recurrence/key -> projection -> low-rank -> shared-input
        // backward cell resident in the timed command stream. The core-only
        // profile performed by `new` remains useful for partial ops, while a
        // full model should select against the cache/descriptor/RAW pressure it
        // will actually see during training.
        op.configure_backward_segment_schedules(true)?;
        Ok(op)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: VulkanDevice,
        width: usize,
        head_size: usize,
        max_batch: usize,
        mix_r: &[f32],
        mix_k: &[f32],
        mix_v: &[f32],
        receptance_weight: &[f32],
        key_weight: &[f32],
        value_weight: &[f32],
        k_k: &[f32],
        k_a: &[f32],
    ) -> Result<Self> {
        if width == 0 || head_size == 0 || max_batch == 0 {
            bail!("RWKV time-mix core width, head_size, and max_batch must be positive");
        }
        if !width.is_multiple_of(head_size) {
            bail!("RWKV width {width} must be divisible by head_size {head_size}");
        }
        validate_len("mix_r", mix_r, width)?;
        validate_len("mix_k", mix_k, width)?;
        validate_len("mix_v", mix_v, width)?;
        validate_len("k_k", k_k, width)?;
        validate_len("k_a", k_a, width)?;
        let weight_len = width
            .checked_mul(width)
            .context("RWKV weight size overflow")?;
        validate_len("receptance_weight", receptance_weight, weight_len)?;
        validate_len("key_weight", key_weight, weight_len)?;
        validate_len("value_weight", value_weight, weight_len)?;

        let heads = width / head_size;
        let vector_len = max_batch
            .checked_mul(width)
            .context("RWKV vector capacity overflow")?;
        let state_len = vector_len
            .checked_mul(head_size)
            .context("RWKV matrix-state capacity overflow")?;

        let forward_workgroup_size = match std::env::var(
            HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE_ENV,
        ) {
            Ok(raw) => match raw.parse::<u32>() {
                Ok(32) => 32,
                Ok(64) => 64,
                _ => bail!(
                    "{HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE_ENV} must be 32 or 64, got {raw:?}"
                ),
            },
            Err(std::env::VarError::NotPresent) => 64,
            Err(err) => bail!(
                "reading {HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE_ENV}: {err}"
            ),
        };
        if forward_workgroup_size < head_size as u32 {
            bail!(
                "{HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE_ENV}={forward_workgroup_size} is smaller than RWKV head_size {head_size}"
            );
        }
        let forward_spirv = if forward_workgroup_size == 32 {
            TIME_MIX_LINEAR3_KEY_STATE_FORWARD_FUSED_WG32_SPV
        } else {
            TIME_MIX_LINEAR3_KEY_STATE_FORWARD_FUSED_SPV
        };
        let time_mix_linear3_key_state_forward_fused = if head_size <= 64
            && device.supports_storage_buffer_bindings(24)
            && device.supports_compute_work_group_size_x(forward_workgroup_size)
        {
            Some(create_time_mix_forward_fused_kernel(
                &device,
                forward_spirv,
            )?)
        } else {
            None
        };
        let time_mix_linear3_key_state_forward_fused_two_rows = if width <= 32
            && head_size <= 32
            && device.supports_storage_buffer_bindings(24)
            && device.supports_compute_work_group_size_x(64)
            && std::env::var_os(HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE_ENV).is_none()
            && std::env::var_os(HIERARCHOS_RWKV_TIME_MIX_ENABLE_TWO_ROW_FORWARD_FUSION_ENV)
                .is_some()
        {
            Some(create_time_mix_forward_fused_kernel(
                &device,
                TIME_MIX_LINEAR3_KEY_STATE_FORWARD_FUSED_TWO_ROWS_SPV,
            )?)
        } else {
            None
        };
        let weight_reuse2_enabled =
            std::env::var_os(HIERARCHOS_RWKV_TIME_MIX_ENABLE_WEIGHT_REUSE2_ENV).is_some();
        let create_full_forward_weight_reuse_set =
            |rows: u32, shared_bytes: [u32; 3], spirv: [&[u8]; 3]| -> Result<WeightReuseKernels> {
                if !weight_reuse2_enabled
                    || head_size > 64
                    || forward_workgroup_size != 64
                    || !device.supports_storage_buffer_bindings(24)
                    || !device.supports_compute_work_group_size_x(64 * rows)
                {
                    return Ok(WeightReuseKernels::default());
                }
                let shared_limit = device.max_compute_shared_memory_bytes();
                Ok(WeightReuseKernels {
                    tile16: if shared_limit >= shared_bytes[0] {
                        Some(create_time_mix_forward_fused_kernel(&device, spirv[0])?)
                    } else {
                        None
                    },
                    tile32: if shared_limit >= shared_bytes[1] {
                        Some(create_time_mix_forward_fused_kernel(&device, spirv[1])?)
                    } else {
                        None
                    },
                    tile64: if shared_limit >= shared_bytes[2] {
                        Some(create_time_mix_forward_fused_kernel(&device, spirv[2])?)
                    } else {
                        None
                    },
                })
            };
        let time_mix_linear3_key_state_forward_weight_reuse = MultiRowWeightReuseKernels {
            rows2: create_full_forward_weight_reuse_set(
                2,
                [
                    TIME_MIX_WEIGHT_REUSE2_TILE16_SHARED_BYTES,
                    TIME_MIX_WEIGHT_REUSE2_TILE32_SHARED_BYTES,
                    TIME_MIX_WEIGHT_REUSE2_TILE64_SHARED_BYTES,
                ],
                [
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE2_SPV,
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE2_TILE32_SPV,
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE2_TILE64_SPV,
                ],
            )?,
            rows4: create_full_forward_weight_reuse_set(
                4,
                [
                    TIME_MIX_WEIGHT_REUSE4_TILE16_SHARED_BYTES,
                    TIME_MIX_WEIGHT_REUSE4_TILE32_SHARED_BYTES,
                    TIME_MIX_WEIGHT_REUSE4_TILE64_SHARED_BYTES,
                ],
                [
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE4_SPV,
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE4_TILE32_SPV,
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE4_TILE64_SPV,
                ],
            )?,
            rows8: create_full_forward_weight_reuse_set(
                8,
                [
                    TIME_MIX_WEIGHT_REUSE8_TILE16_SHARED_BYTES,
                    TIME_MIX_WEIGHT_REUSE8_TILE32_SHARED_BYTES,
                    TIME_MIX_WEIGHT_REUSE8_TILE64_SHARED_BYTES,
                ],
                [
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE8_SPV,
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE8_TILE32_SPV,
                    TIME_MIX_LINEAR3_KEY_STATE_FORWARD_WEIGHT_REUSE8_TILE64_SPV,
                ],
            )?,
        };
        let packed_fast_spirv = if forward_workgroup_size == 32 {
            TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WG32_SPV
        } else {
            TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_SPV
        };
        let time_mix_linear3_key_state_forward_packed_fast = if head_size <= 64
            && device.supports_storage_buffer_bindings(18)
            && device.supports_compute_work_group_size_x(forward_workgroup_size)
        {
            Some(create_time_mix_forward_packed_fast_kernel(
                &device,
                packed_fast_spirv,
            )?)
        } else {
            None
        };
        let packed_tape_spirv = if forward_workgroup_size == 32 {
            TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_TAPE_WG32_SPV
        } else {
            TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_TAPE_SPV
        };
        let time_mix_linear3_key_state_forward_packed_tape = if head_size <= 64
            && device.supports_storage_buffer_bindings(24)
            && device.supports_compute_work_group_size_x(forward_workgroup_size)
        {
            Some(create_time_mix_forward_packed_tape_kernel(
                &device,
                packed_tape_spirv,
            )?)
        } else {
            None
        };
        let create_packed_forward_weight_reuse_set =
            |rows: u32, shared_bytes: [u32; 3], spirv: [&[u8]; 3]| -> Result<WeightReuseKernels> {
                if !weight_reuse2_enabled
                    || head_size > 64
                    || forward_workgroup_size != 64
                    || !device.supports_storage_buffer_bindings(18)
                    || !device.supports_compute_work_group_size_x(64 * rows)
                {
                    return Ok(WeightReuseKernels::default());
                }
                let shared_limit = device.max_compute_shared_memory_bytes();
                Ok(WeightReuseKernels {
                    tile16: if shared_limit >= shared_bytes[0] {
                        Some(create_time_mix_forward_packed_fast_kernel(
                            &device, spirv[0],
                        )?)
                    } else {
                        None
                    },
                    tile32: if shared_limit >= shared_bytes[1] {
                        Some(create_time_mix_forward_packed_fast_kernel(
                            &device, spirv[1],
                        )?)
                    } else {
                        None
                    },
                    tile64: if shared_limit >= shared_bytes[2] {
                        Some(create_time_mix_forward_packed_fast_kernel(
                            &device, spirv[2],
                        )?)
                    } else {
                        None
                    },
                })
            };
        let time_mix_linear3_key_state_forward_packed_fast_weight_reuse =
            MultiRowWeightReuseKernels {
                rows2: create_packed_forward_weight_reuse_set(
                    2,
                    [
                        TIME_MIX_WEIGHT_REUSE2_TILE16_SHARED_BYTES,
                        TIME_MIX_WEIGHT_REUSE2_TILE32_SHARED_BYTES,
                        TIME_MIX_WEIGHT_REUSE2_TILE64_SHARED_BYTES,
                    ],
                    [
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE2_SPV,
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE2_TILE32_SPV,
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE2_TILE64_SPV,
                    ],
                )?,
                rows4: create_packed_forward_weight_reuse_set(
                    4,
                    [
                        TIME_MIX_WEIGHT_REUSE4_TILE16_SHARED_BYTES,
                        TIME_MIX_WEIGHT_REUSE4_TILE32_SHARED_BYTES,
                        TIME_MIX_WEIGHT_REUSE4_TILE64_SHARED_BYTES,
                    ],
                    [
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE4_SPV,
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE4_TILE32_SPV,
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE4_TILE64_SPV,
                    ],
                )?,
                rows8: create_packed_forward_weight_reuse_set(
                    8,
                    [
                        TIME_MIX_WEIGHT_REUSE8_TILE16_SHARED_BYTES,
                        TIME_MIX_WEIGHT_REUSE8_TILE32_SHARED_BYTES,
                        TIME_MIX_WEIGHT_REUSE8_TILE64_SHARED_BYTES,
                    ],
                    [
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE8_SPV,
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE8_TILE32_SPV,
                        TIME_MIX_LINEAR3_KEY_STATE_FORWARD_PACKED_FAST_WEIGHT_REUSE8_TILE64_SPV,
                    ],
                )?,
            };

        let linear3_time_mix_backward_fused = if device.supports_storage_buffer_bindings(16) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR3_TIME_MIX_BACKWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<MixPush>() as u32,
            )?)
        } else {
            None
        };

        let state_backward_fused_rkv_add3_wg32 = create_state_backward_geometry_kernel(
            &device,
            RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_WG32_SPV,
            22,
            32,
        )?;
        let state_backward_fused_rkv_add3_wg128 = create_state_backward_geometry_kernel(
            &device,
            RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_WG128_SPV,
            22,
            128,
        )?;
        let state_backward_fused_rkv_add3_key_transform_wg32 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_WG32_SPV,
                29,
                32,
            )?;
        let state_backward_fused_rkv_add3_key_transform_wg128 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_WG128_SPV,
                29,
                128,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_wg32 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_WG32_SPV,
                31,
                32,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_wg128 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_WG128_SPV,
                31,
                128,
            )?;

        let subgroup_backward_available = device.supports_compute_subgroup_arithmetic();
        let subgroup_recurrent_available = subgroup_backward_available;
        let state_backward_fused_rkv_add3_recurrent_subgroup = if subgroup_recurrent_available {
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_RECURRENT_SUBGROUP_SPV,
                22,
                64,
            )?
        } else {
            None
        };
        let state_backward_fused_rkv_add3_recurrent_subgroup_wg32 = if subgroup_recurrent_available
        {
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_RECURRENT_SUBGROUP_WG32_SPV,
                22,
                32,
            )?
        } else {
            None
        };
        let state_backward_fused_rkv_add3_recurrent_subgroup_wg128 = if subgroup_recurrent_available
        {
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_RECURRENT_SUBGROUP_WG128_SPV,
                22,
                128,
            )?
        } else {
            None
        };
        let key_transform_backward_subgroup = if subgroup_backward_available
            && device.supports_compute_work_group_size_x(64)
            && device.supports_storage_buffer_bindings(12)
        {
            Some(vulkan::ComputeKernel::new(
                &device,
                KEY_TRANSFORM_BACKWARD_SUBGROUP_SPV,
                12,
                std::mem::size_of::<KeyPush>() as u32,
            )?)
        } else {
            None
        };
        let state_backward_fused_rkv_add3_key_transform_subgroup = if subgroup_backward_available {
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SUBGROUP_SPV,
                29,
                64,
            )?
        } else {
            None
        };
        let state_backward_fused_rkv_add3_key_transform_subgroup_wg32 =
            if subgroup_backward_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SUBGROUP_WG32_SPV,
                    29,
                    32,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_subgroup_wg128 =
            if subgroup_backward_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SUBGROUP_WG128_SPV,
                    29,
                    128,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_recurrent_subgroup =
            if subgroup_recurrent_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_RECURRENT_SUBGROUP_SPV,
                    29,
                    64,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg32 =
            if subgroup_recurrent_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_RECURRENT_SUBGROUP_WG32_SPV,
                    29,
                    32,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg128 =
            if subgroup_recurrent_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_RECURRENT_SUBGROUP_WG128_SPV,
                    29,
                    128,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_reduce_subgroup =
            if subgroup_backward_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SUBGROUP_SPV,
                    31,
                    64,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg32 =
            if subgroup_backward_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SUBGROUP_WG32_SPV,
                    31,
                    32,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg128 =
            if subgroup_backward_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SUBGROUP_WG128_SPV,
                    31,
                    128,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup =
            if subgroup_recurrent_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_RECURRENT_SUBGROUP_SPV,
                    31,
                    64,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg32 =
            if subgroup_recurrent_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_RECURRENT_SUBGROUP_WG32_SPV,
                    31,
                    32,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg128 =
            if subgroup_recurrent_available {
                create_state_backward_geometry_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_RECURRENT_SUBGROUP_WG128_SPV,
                    31,
                    128,
                )?
            } else {
                None
            };
        let state_backward_fused_rkv_add3_key_transform_tree =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TREE_SPV,
                29,
                64,
            )?;
        let state_backward_fused_rkv_add3_key_transform_tree_wg32 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TREE_WG32_SPV,
                29,
                32,
            )?;
        let state_backward_fused_rkv_add3_key_transform_tree_wg128 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TREE_WG128_SPV,
                29,
                128,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_tree =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TREE_SPV,
                31,
                64,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_tree_wg32 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TREE_WG32_SPV,
                31,
                32,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_tree_wg128 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TREE_WG128_SPV,
                31,
                128,
            )?;
        let state_backward_fused_rkv_add3_key_transform_tiled =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TILED_SPV,
                29,
                64,
            )?;
        let state_backward_fused_rkv_add3_key_transform_tiled_wg32 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TILED_WG32_SPV,
                29,
                32,
            )?;
        let state_backward_fused_rkv_add3_key_transform_tiled_wg128 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_TILED_WG128_SPV,
                29,
                128,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_tiled =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TILED_SPV,
                31,
                64,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg32 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TILED_WG32_SPV,
                31,
                32,
            )?;
        let state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg128 =
            create_state_backward_geometry_kernel(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_TILED_WG128_SPV,
                31,
                128,
            )?;

        let projection_input_grad_reuse_supported = device.supports_storage_buffer_bindings(9);
        let create_projection_input_grad_reuse = |spirv: &[u8]| {
            vulkan::ComputeKernel::new_with_access(
                &device,
                spirv,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )
        };
        let create_projection_input_grad_reuse_set = |workgroup_size: u32,
                                                      spirv16: &[u8],
                                                      spirv32: &[u8],
                                                      spirv64: &[u8]|
         -> Result<WeightReuseKernels> {
            if !projection_input_grad_reuse_supported
                || !device.supports_compute_work_group_size_x(workgroup_size)
            {
                return Ok(WeightReuseKernels::default());
            }
            Ok(WeightReuseKernels {
                tile16: if device.max_compute_shared_memory_bytes()
                    >= LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE16_SHARED_BYTES
                {
                    Some(create_projection_input_grad_reuse(spirv16)?)
                } else {
                    None
                },
                tile32: if device.max_compute_shared_memory_bytes()
                    >= LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE32_SHARED_BYTES
                {
                    Some(create_projection_input_grad_reuse(spirv32)?)
                } else {
                    None
                },
                tile64: if device.max_compute_shared_memory_bytes()
                    >= LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE64_SHARED_BYTES
                {
                    Some(create_projection_input_grad_reuse(spirv64)?)
                } else {
                    None
                },
            })
        };
        let linear3_input_grad_weight_reuse = MultiRowWeightReuseKernels {
            rows2: create_projection_input_grad_reuse_set(
                128,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_SPV,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE32_SPV,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE2_TILE64_SPV,
            )?,
            rows4: create_projection_input_grad_reuse_set(
                256,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE4_SPV,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE4_TILE32_SPV,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE4_TILE64_SPV,
            )?,
            rows8: create_projection_input_grad_reuse_set(
                512,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE8_SPV,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE8_TILE32_SPV,
                LINEAR3_INPUT_GRAD_WEIGHT_REUSE8_TILE64_SPV,
            )?,
        };

        let mut op = Self {
            time_mix_forward: vulkan::ComputeKernel::new(
                &device,
                TIME_MIX_FORWARD_SPV,
                8,
                std::mem::size_of::<MixPush>() as u32,
            )?,
            time_mix_backward: vulkan::ComputeKernel::new(
                &device,
                TIME_MIX_BACKWARD_SPV,
                13,
                std::mem::size_of::<MixPush>() as u32,
            )?,
            linear3_forward: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR3_FORWARD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            time_mix_linear3_key_state_forward_fused,
            time_mix_linear3_key_state_forward_fused_two_rows,
            time_mix_linear3_key_state_forward_weight_reuse,
            time_mix_linear3_key_state_forward_packed_fast,
            time_mix_linear3_key_state_forward_packed_fast_weight_reuse,
            time_mix_linear3_key_state_forward_packed_tape,
            linear3_weight_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR3_WEIGHT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear3_weight_grad_tiled: if device.supports_compute_work_group_size([16, 16, 1])
                && device.max_compute_shared_memory_bytes() >= 6_144
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    LINEAR3_WEIGHT_GRAD_TILED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            linear3_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR3_INPUT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear3_input_grad_weight_reuse,
            linear3_input_grad_fp16_scaled: if device
                .mixed_precision_capabilities()
                .shader_float16_enabled
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    LINEAR3_INPUT_GRAD_FP16_SCALED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            linear3_input_grad_fp16_source_scaled: if device
                .mixed_precision_capabilities()
                .shader_float16_enabled
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    LINEAR3_INPUT_GRAD_FP16_SOURCE_SCALED_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<LinearPush>() as u32,
                )?)
            } else {
                None
            },
            native_fp16_projection_input_grad: false,
            backward_source_scale: 1.0,
            source_scaled_backward_domain: false,
            linear3_time_mix_backward_fused,
            key_transform_forward: vulkan::ComputeKernel::new(
                &device,
                KEY_TRANSFORM_FORWARD_SPV,
                6,
                std::mem::size_of::<KeyPush>() as u32,
            )?,
            key_transform_backward: vulkan::ComputeKernel::new(
                &device,
                KEY_TRANSFORM_BACKWARD_SPV,
                12,
                std::mem::size_of::<KeyPush>() as u32,
            )?,
            key_transform_backward_subgroup,
            key_transform_param_reduce: vulkan::ComputeKernel::new(
                &device,
                KEY_TRANSFORM_PARAM_REDUCE_SPV,
                4,
                std::mem::size_of::<MixPush>() as u32,
            )?,
            state_forward: vulkan::ComputeKernel::new(
                &device,
                RWKV_MATRIX_STATE_FORWARD_SPV,
                10,
                std::mem::size_of::<StatePush>() as u32,
            )?,
            key_state_forward_fused: vulkan::ComputeKernel::new_with_access(
                &device,
                RWKV_KEY_STATE_FORWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            state_backward_rows: vulkan::ComputeKernel::new(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_ROWS_SPV,
                11,
                std::mem::size_of::<StatePush>() as u32,
            )?,
            state_backward_cols: vulkan::ComputeKernel::new(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_COLS_SPV,
                17,
                std::mem::size_of::<StatePush>() as u32,
            )?,
            state_backward_fused: vulkan::ComputeKernel::new_with_access(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            state_backward_fused_rk_add: if device.supports_storage_buffer_bindings(20) {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RK_ADD_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<StatePush>() as u32,
                )?)
            } else {
                None
            },
            state_backward_fused_rkv_add3: if device.supports_storage_buffer_bindings(22) {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<StatePush>() as u32,
                )?)
            } else {
                None
            },
            state_backward_fused_rkv_add3_wg32,
            state_backward_fused_rkv_add3_wg128,
            state_backward_fused_rkv_add3_recurrent_subgroup,
            state_backward_fused_rkv_add3_recurrent_subgroup_wg32,
            state_backward_fused_rkv_add3_recurrent_subgroup_wg128,
            state_backward_fused_rkv_add3_key_transform: if device
                .supports_storage_buffer_bindings(29)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<StatePush>() as u32,
                )?)
            } else {
                None
            },
            state_backward_fused_rkv_add3_key_transform_packed:
                create_state_backward_packed_kernel(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_PACKED_SPV,
                )?,
            state_backward_fused_rkv_add3_key_transform_wg32,
            state_backward_fused_rkv_add3_key_transform_wg128,
            state_backward_fused_rkv_add3_key_transform_subgroup,
            state_backward_fused_rkv_add3_key_transform_subgroup_wg32,
            state_backward_fused_rkv_add3_key_transform_subgroup_wg128,
            state_backward_fused_rkv_add3_key_transform_recurrent_subgroup,
            state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg32,
            state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg128,
            state_backward_fused_rkv_add3_key_transform_tree,
            state_backward_fused_rkv_add3_key_transform_tree_wg32,
            state_backward_fused_rkv_add3_key_transform_tree_wg128,
            state_backward_fused_rkv_add3_key_transform_tiled,
            state_backward_fused_rkv_add3_key_transform_tiled_wg32,
            state_backward_fused_rkv_add3_key_transform_tiled_wg128,
            state_backward_fused_rkv_add3_key_transform_reduce: if device
                .supports_storage_buffer_bindings(31)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    RWKV_MATRIX_STATE_BACKWARD_FUSED_RKV_ADD3_KEY_TRANSFORM_REDUCE_SPV,
                    &[
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<StatePush>() as u32,
                )?)
            } else {
                None
            },
            state_backward_fused_rkv_add3_key_transform_reduce_wg32,
            state_backward_fused_rkv_add3_key_transform_reduce_wg128,
            state_backward_fused_rkv_add3_key_transform_reduce_subgroup,
            state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg32,
            state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg128,
            state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup,
            state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg32,
            state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg128,
            state_backward_fused_rkv_add3_key_transform_reduce_tree,
            state_backward_fused_rkv_add3_key_transform_reduce_tree_wg32,
            state_backward_fused_rkv_add3_key_transform_reduce_tree_wg128,
            state_backward_fused_rkv_add3_key_transform_reduce_tiled,
            state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg32,
            state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg128,
            numerics_policy: RwkvNumericsPolicy::StrictParity,
            state_backward_schedule_batch1: None,
            state_backward_schedule_multi: None,
            state_backward_profile_batch: max_batch.min(8),
            backward_segment_schedule_batch1: None,
            backward_segment_schedule_multi: None,
            backward_kernel_geometry_batch1: BackwardKernelGeometry::Wg64,
            backward_kernel_geometry_multi: BackwardKernelGeometry::Wg64,
            vector_add: vulkan::ComputeKernel::new_with_access(
                &device,
                VECTOR_ADD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            vector_add3: vulkan::ComputeKernel::new_with_access(
                &device,
                VECTOR_ADD3_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            mix_r: GpuBuffer::from_f32(&device, mix_r)?,
            mix_k: GpuBuffer::from_f32(&device, mix_k)?,
            mix_v: GpuBuffer::from_f32(&device, mix_v)?,
            receptance_weight: GpuBuffer::from_f32(&device, receptance_weight)?,
            key_weight: GpuBuffer::from_f32(&device, key_weight)?,
            value_weight: GpuBuffer::from_f32(&device, value_weight)?,
            k_k: GpuBuffer::from_f32(&device, k_k)?,
            k_a: GpuBuffer::from_f32(&device, k_a)?,
            state: GpuBuffer::zeros_f32(&device, state_len)?,
            x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            a: GpuBuffer::zeros_f32(&device, vector_len)?,
            w: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_new_state: GpuBuffer::zeros_f32(&device, state_len)?,
            grad_tmix: GpuBuffer::zeros_f32(&device, vector_len)?,
            xr: GpuBuffer::zeros_f32(&device, vector_len)?,
            xk: GpuBuffer::zeros_f32(&device, vector_len)?,
            xv: GpuBuffer::zeros_f32(&device, vector_len)?,
            r: GpuBuffer::zeros_f32(&device, vector_len)?,
            raw_k: GpuBuffer::zeros_f32(&device, vector_len)?,
            v: GpuBuffer::zeros_f32(&device, vector_len)?,
            scaled_k: GpuBuffer::zeros_f32(&device, vector_len)?,
            kk: GpuBuffer::zeros_f32(&device, vector_len)?,
            new_state: GpuBuffer::zeros_f32(&device, state_len)?,
            tmix: GpuBuffer::zeros_f32(&device, vector_len)?,
            saved_sa: GpuBuffer::zeros_f32(&device, vector_len)?,
            saved_q: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_state: GpuBuffer::zeros_f32(&device, state_len)?,
            grad_r: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_scaled_k: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_v: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_kk: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_a_direct: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_w: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_raw_k: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_a: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_k_k_partial: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_k_a_partial: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_k_k: GpuBuffer::zeros_f32(&device, width)?,
            grad_k_a: GpuBuffer::zeros_f32(&device, width)?,
            grad_xr: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_xk: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_xv: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_receptance_weight: GpuBuffer::zeros_f32(&device, weight_len)?,
            grad_key_weight: GpuBuffer::zeros_f32(&device, weight_len)?,
            grad_value_weight: GpuBuffer::zeros_f32(&device, weight_len)?,
            grad_x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_mix_r: GpuBuffer::zeros_f32(&device, width)?,
            grad_mix_k: GpuBuffer::zeros_f32(&device, width)?,
            grad_mix_v: GpuBuffer::zeros_f32(&device, width)?,
            total_grad_r: GpuBuffer::zeros_f32(&device, vector_len)?,
            total_grad_scaled_k: GpuBuffer::zeros_f32(&device, vector_len)?,
            total_grad_v: GpuBuffer::zeros_f32(&device, vector_len)?,
            total_grad_v_external: GpuBuffer::zeros_f32(&device, vector_len)?,
            fused_grad_x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            fused_grad_previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_g_external: GpuBuffer::zeros_f32(&device, vector_len)?,
            new_state_readback: GpuBuffer::zeros_host_f32(&device, state_len)?,
            tmix_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            scaled_k_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            kk_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_state_readback: GpuBuffer::zeros_host_f32(&device, state_len)?,
            grad_x_norm_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_previous_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_a_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_w_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_mix_r_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_mix_k_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_mix_v_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_receptance_weight_readback: GpuBuffer::zeros_host_f32(&device, weight_len)?,
            grad_key_weight_readback: GpuBuffer::zeros_host_f32(&device, weight_len)?,
            grad_value_weight_readback: GpuBuffer::zeros_host_f32(&device, weight_len)?,
            grad_k_k_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_k_a_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            fused_grad_x_norm_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            fused_grad_previous_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            device,
            width,
            head_size,
            heads,
            max_batch,
            low_rank: None,
            post_mix: None,
        };
        op.configure_state_backward_schedules()?;
        op.configure_backward_segment_schedules(false)?;
        Ok(op)
    }

    fn record_projection_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        linear_push: &LinearPush,
        groups: [u32; 3],
    ) -> Result<()> {
        self.linear3_forward.record_dispatch(
            commands,
            &[
                &self.xr,
                &self.receptance_weight,
                &self.xk,
                &self.key_weight,
                &self.xv,
                &self.value_weight,
                &self.r,
                &self.raw_k,
                &self.v,
            ],
            bytemuck::bytes_of(linear_push),
            groups,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_time_mix_projection_key_state_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        mix_push: &MixPush,
        linear_push: &LinearPush,
        key_push: &KeyPush,
        state_push: &StatePush,
        linear_forward_groups: [u32; 3],
        head_groups: [u32; 3],
        vector_groups: [u32; 3],
    ) -> Result<()> {
        let decision = self.choose_full_forward_projection_topology(batch)?;
        self.record_time_mix_projection_key_state_forward_topology(
            commands,
            batch,
            state,
            x_norm,
            previous,
            a,
            w,
            mix_push,
            linear_push,
            key_push,
            state_push,
            linear_forward_groups,
            head_groups,
            vector_groups,
            decision.topology,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_time_mix_projection_key_state_forward_topology(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        mix_push: &MixPush,
        linear_push: &LinearPush,
        key_push: &KeyPush,
        state_push: &StatePush,
        linear_forward_groups: [u32; 3],
        head_groups: [u32; 3],
        vector_groups: [u32; 3],
        topology: ForwardProjectionTopology,
    ) -> Result<()> {
        if let ForwardProjectionTopology::WeightReuse { rows, tile } = topology {
            if batch < rows as usize {
                bail!("RWKV weight-reuse{rows} forward topology requires batch >= {rows}");
            }
            if let Some(kernel) = self
                .time_mix_linear3_key_state_forward_weight_reuse
                .get(rows, tile)
            {
                let groups = self
                    .heads
                    .checked_mul(batch.div_ceil(rows as usize))
                    .context("RWKV weight-reuse forward workgroup count overflow")?;
                let groups = u32::try_from(groups).context(
                    "RWKV weight-reuse forward workgroup count exceeds Vulkan u32 range",
                )?;
                return kernel.record_dispatch(
                    commands,
                    &[
                        x_norm,
                        previous,
                        &self.mix_r,
                        &self.mix_k,
                        &self.mix_v,
                        &self.receptance_weight,
                        &self.key_weight,
                        &self.value_weight,
                        a,
                        &self.k_k,
                        &self.k_a,
                        state,
                        w,
                        &self.xr,
                        &self.xk,
                        &self.xv,
                        &self.r,
                        &self.raw_k,
                        &self.v,
                        &self.scaled_k,
                        &self.kk,
                        &self.new_state,
                        &self.tmix,
                        &self.saved_sa,
                    ],
                    bytemuck::bytes_of(state_push),
                    [groups, 1, 1],
                );
            }
            bail!(
                "RWKV weight-reuse{rows} forward tile {tile} is unavailable on this Vulkan device"
            );
        }

        let (fused_kernel, fused_two_rows) =
            if let Some(kernel) = &self.time_mix_linear3_key_state_forward_fused_two_rows {
                (Some(kernel), true)
            } else {
                (
                    self.time_mix_linear3_key_state_forward_fused.as_ref(),
                    false,
                )
            };
        if let Some(kernel) = fused_kernel {
            let linear_heads = batch
                .checked_mul(self.heads)
                .context("RWKV fused time-mix/projection/state workgroup count overflow")?;
            let groups = u32::try_from(if fused_two_rows {
                linear_heads.div_ceil(2)
            } else {
                linear_heads
            })
            .context(
                "RWKV fused time-mix/projection/state workgroup count exceeds Vulkan u32 range",
            )?;
            return kernel.record_dispatch(
                commands,
                &[
                    x_norm,
                    previous,
                    &self.mix_r,
                    &self.mix_k,
                    &self.mix_v,
                    &self.receptance_weight,
                    &self.key_weight,
                    &self.value_weight,
                    a,
                    &self.k_k,
                    &self.k_a,
                    state,
                    w,
                    &self.xr,
                    &self.xk,
                    &self.xv,
                    &self.r,
                    &self.raw_k,
                    &self.v,
                    &self.scaled_k,
                    &self.kk,
                    &self.new_state,
                    &self.tmix,
                    &self.saved_sa,
                ],
                bytemuck::bytes_of(state_push),
                [groups, 1, 1],
            );
        }

        self.time_mix_forward.record_dispatch(
            commands,
            &[
                x_norm,
                previous,
                &self.mix_r,
                &self.mix_k,
                &self.mix_v,
                &self.xr,
                &self.xk,
                &self.xv,
            ],
            bytemuck::bytes_of(mix_push),
            vector_groups,
        )?;
        self.record_projection_forward(commands, linear_push, linear_forward_groups)?;
        self.record_key_state_forward(
            commands,
            batch,
            state,
            a,
            w,
            key_push,
            state_push,
            head_groups,
            vector_groups,
        )
    }

    fn available_forward_projection_topologies(
        &self,
        recurrence: ForwardProjectionRecurrence,
        batch: usize,
    ) -> Vec<ForwardProjectionTopology> {
        let mut candidates = Vec::with_capacity(4);
        match recurrence {
            ForwardProjectionRecurrence::Full => {
                candidates.push(ForwardProjectionTopology::Baseline)
            }
            ForwardProjectionRecurrence::Packed => {
                if self
                    .time_mix_linear3_key_state_forward_packed_fast
                    .is_some()
                {
                    candidates.push(ForwardProjectionTopology::Baseline);
                }
            }
        }
        let kernels = match recurrence {
            ForwardProjectionRecurrence::Full => {
                &self.time_mix_linear3_key_state_forward_weight_reuse
            }
            ForwardProjectionRecurrence::Packed => {
                &self.time_mix_linear3_key_state_forward_packed_fast_weight_reuse
            }
        };
        for rows in [2u32, 4, 8] {
            if batch < rows as usize {
                continue;
            }
            let Some(row_kernels) = kernels.rows(rows) else {
                continue;
            };
            candidates.extend(
                row_kernels
                    .available_tiles()
                    .into_iter()
                    .map(|tile| ForwardProjectionTopology::WeightReuse { rows, tile }),
            );
        }
        candidates
    }

    fn structural_forward_projection_topology(
        &self,
        recurrence: ForwardProjectionRecurrence,
        batch: usize,
    ) -> Option<ForwardProjectionTopology> {
        let kernels = match recurrence {
            ForwardProjectionRecurrence::Full => {
                &self.time_mix_linear3_key_state_forward_weight_reuse
            }
            ForwardProjectionRecurrence::Packed => {
                &self.time_mix_linear3_key_state_forward_packed_fast_weight_reuse
            }
        };
        for rows in [8u32, 4, 2] {
            if batch < rows as usize {
                continue;
            }
            if let Some(tile) = kernels
                .rows(rows)
                .and_then(WeightReuseKernels::largest_tile)
            {
                return Some(ForwardProjectionTopology::WeightReuse { rows, tile });
            }
        }
        self.available_forward_projection_topologies(recurrence, batch)
            .into_iter()
            .find(|candidate| *candidate == ForwardProjectionTopology::Baseline)
    }

    fn choose_forward_projection_topology<F>(
        &self,
        batch: usize,
        recurrence: ForwardProjectionRecurrence,
        mut time_topology_ms: F,
    ) -> Result<ForwardProjectionDecision>
    where
        F: FnMut(ForwardProjectionTopology) -> Result<f64>,
    {
        let candidates = self.available_forward_projection_topologies(recurrence, batch);
        let baseline = candidates
            .iter()
            .copied()
            .find(|candidate| *candidate == ForwardProjectionTopology::Baseline)
            .or_else(|| candidates.first().copied())
            .context("RWKV forward projection has no compatible topology")?;

        if std::env::var_os(HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_DISABLE_AUTOTUNE_ENV).is_some() {
            return Ok(ForwardProjectionDecision {
                topology: self
                    .structural_forward_projection_topology(recurrence, batch)
                    .unwrap_or(baseline),
                autotuned: false,
            });
        }
        if candidates.len() == 1 {
            return Ok(ForwardProjectionDecision {
                topology: candidates[0],
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let key = ForwardProjectionAutotuneKey {
            device_name: self.device.name().to_owned(),
            subgroup_size,
            width: self.width,
            head_size: self.head_size,
            batch_pairs: batch.div_ceil(2),
            has_unpaired_tail: batch % 2 != 0,
            recurrence,
        };
        let cache = FORWARD_PROJECTION_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(decision) = cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV forward-projection autotune cache lock was poisoned")
            })?
            .get(&key)
            .copied()
        {
            return Ok(decision);
        }

        // Prime the device with the portable anchor before interleaving the
        // measured candidates. Forward projection kernels are short enough for
        // DVFS/submit noise to otherwise dominate the tile decision.
        if let Err(err) = time_topology_ms(baseline) {
            if std::env::var_os(HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_AUTOTUNE_LOG_ENV).is_some() {
                eprintln!(
                    "RWKV forward-projection autotune warmup failed device={} recurrence={} batch={batch}: {err:#}; using {}",
                    self.device.name(),
                    recurrence.label(),
                    baseline.label()
                );
            }
            return Ok(ForwardProjectionDecision {
                topology: baseline,
                autotuned: false,
            });
        }

        let reuse_coordinates = candidates
            .iter()
            .filter_map(|topology| match *topology {
                ForwardProjectionTopology::Baseline => None,
                ForwardProjectionTopology::WeightReuse { rows, tile } => {
                    Some(ProjectionReuseCoordinate { rows, tile })
                }
            })
            .collect::<Vec<_>>();
        let preferred_reuse = self
            .structural_forward_projection_topology(recurrence, batch)
            .and_then(|topology| match topology {
                ForwardProjectionTopology::Baseline => None,
                ForwardProjectionTopology::WeightReuse { rows, tile } => {
                    Some(ProjectionReuseCoordinate { rows, tile })
                }
            });
        let reuse_probe_coordinates =
            factorized_projection_reuse_probe_coordinates(&reuse_coordinates, preferred_reuse);
        let mut probe_candidates = Vec::with_capacity(reuse_probe_coordinates.len() + 1);
        probe_candidates.push(baseline);
        for coordinate in &reuse_probe_coordinates {
            let topology = ForwardProjectionTopology::WeightReuse {
                rows: coordinate.rows,
                tile: coordinate.tile,
            };
            if candidates.contains(&topology) && !probe_candidates.contains(&topology) {
                probe_candidates.push(topology);
            }
        }
        let mut timings = match median_interleaved_timings(&probe_candidates, |topology| {
            time_topology_ms(topology).with_context(|| {
                format!(
                    "timing RWKV forward projection candidate {}",
                    topology.label()
                )
            })
        }) {
            Ok(timings) => timings,
            Err(err) => {
                if std::env::var_os(HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_AUTOTUNE_LOG_ENV)
                    .is_some()
                {
                    eprintln!(
                        "RWKV forward-projection factorized autotune failed device={} recurrence={} batch={batch}: {err:#}; using {}",
                        self.device.name(),
                        recurrence.label(),
                        baseline.label()
                    );
                }
                return Ok(ForwardProjectionDecision {
                    topology: baseline,
                    autotuned: false,
                });
            }
        };
        let reuse_timings = timings
            .iter()
            .filter_map(|(topology, ms)| match *topology {
                ForwardProjectionTopology::Baseline => None,
                ForwardProjectionTopology::WeightReuse { rows, tile } => {
                    Some((ProjectionReuseCoordinate { rows, tile }, *ms))
                }
            })
            .collect::<Vec<_>>();
        let synthesized_reuse = reuse_probe_coordinates.first().copied().and_then(|anchor| {
            select_factorized_projection_reuse_coordinate(
                &reuse_coordinates,
                &reuse_timings,
                anchor,
            )
        });
        let synthesized_was_unmeasured = synthesized_reuse.is_some_and(|coordinate| {
            !reuse_timings
                .iter()
                .any(|(measured, _)| *measured == coordinate)
        });
        if let Some(coordinate) = synthesized_reuse.filter(|coordinate| {
            !reuse_timings
                .iter()
                .any(|(measured, _)| measured == coordinate)
        }) {
            let topology = ForwardProjectionTopology::WeightReuse {
                rows: coordinate.rows,
                tile: coordinate.tile,
            };
            match median_interleaved_timings(&[topology], |candidate| {
                time_topology_ms(candidate).with_context(|| {
                    format!(
                        "timing synthesized RWKV forward projection candidate {}",
                        candidate.label()
                    )
                })
            }) {
                Ok(mut synthesized_timing) => timings.append(&mut synthesized_timing),
                Err(err) => {
                    if std::env::var_os(HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_AUTOTUNE_LOG_ENV)
                        .is_some()
                    {
                        eprintln!(
                            "RWKV forward-projection synthesized factorized candidate failed device={} recurrence={} batch={batch} candidate={}: {err:#}; using {}",
                            self.device.name(),
                            recurrence.label(),
                            topology.label(),
                            baseline.label()
                        );
                    }
                    return Ok(ForwardProjectionDecision {
                        topology: baseline,
                        autotuned: false,
                    });
                }
            }
        }
        let selected = select_projection_traffic_topology(&timings, baseline);
        let decision = ForwardProjectionDecision {
            topology: selected,
            autotuned: true,
        };

        if std::env::var_os(HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_AUTOTUNE_LOG_ENV).is_some() {
            let summary = timings
                .iter()
                .map(|(topology, ms)| format!("{}={ms:.4}ms", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV forward-projection autotune device={} subgroup={} width={} head_size={} batch_pairs={} tail={} recurrence={} policy=factorized-rows-tile measured_reuse_arms={} full_reuse_arms={} synthesized_unmeasured={} {} selected={} autotuned={}",
                self.device.name(),
                subgroup_size,
                self.width,
                self.head_size,
                key.batch_pairs,
                key.has_unpaired_tail,
                recurrence.label(),
                timings.len().saturating_sub(1),
                reuse_coordinates.len(),
                synthesized_was_unmeasured,
                summary,
                selected.label(),
                decision.autotuned
            );
        }

        cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV forward-projection autotune cache lock was poisoned")
            })?
            .insert(key, decision);
        Ok(decision)
    }

    fn choose_full_forward_projection_topology(
        &self,
        batch: usize,
    ) -> Result<ForwardProjectionDecision> {
        self.choose_forward_projection_topology(
            batch,
            ForwardProjectionRecurrence::Full,
            |topology| self.time_full_forward_projection_topology_ms(batch, topology),
        )
    }

    fn time_full_forward_projection_topology_ms(
        &self,
        batch: usize,
        topology: ForwardProjectionTopology,
    ) -> Result<f64> {
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV forward-projection autotune vector size overflow")?;
        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let key_push = KeyPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let vector_groups = [div_ceil_u32(vector_len, 64), 1, 1];
        let head_groups = [div_ceil_u32(batch * self.heads, 64), 1, 1];
        let linear_forward_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let repetitions = if batch.saturating_mul(self.heads) >= 128 {
            4
        } else {
            16
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..repetitions {
            self.record_time_mix_projection_key_state_forward_topology(
                &mut commands,
                batch,
                &self.state,
                &self.x_norm,
                &self.previous,
                &self.a,
                &self.w,
                &mix_push,
                &linear_push,
                &key_push,
                &state_push,
                linear_forward_groups,
                head_groups,
                vector_groups,
                topology,
            )?;
        }
        let started = Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    fn record_projection_weight_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        grad_r: &GpuBuffer,
        grad_k: &GpuBuffer,
        grad_v: &GpuBuffer,
        linear_push: &LinearPush,
        weight_groups: [u32; 3],
    ) -> Result<()> {
        let decision = self.choose_projection_weight_grad_topology(linear_push.rows as usize)?;
        self.record_projection_weight_grad_topology_with_inputs(
            commands,
            &self.xr,
            &self.xk,
            &self.xv,
            grad_r,
            grad_k,
            grad_v,
            linear_push,
            weight_groups,
            decision.topology,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_projection_weight_grad_topology_with_inputs(
        &self,
        commands: &mut vulkan::ComputeBatch,
        xr: &GpuBuffer,
        xk: &GpuBuffer,
        xv: &GpuBuffer,
        grad_r: &GpuBuffer,
        grad_k: &GpuBuffer,
        grad_v: &GpuBuffer,
        linear_push: &LinearPush,
        weight_groups: [u32; 3],
        topology: ProjectionWeightGradTopology,
    ) -> Result<()> {
        let kernel = match topology {
            ProjectionWeightGradTopology::Baseline => &self.linear3_weight_grad,
            ProjectionWeightGradTopology::Tiled => self
                .linear3_weight_grad_tiled
                .as_ref()
                .context("RWKV tiled projection weight-gradient kernel is unavailable")?,
        };
        kernel.record_dispatch(
            commands,
            &[
                xr,
                grad_r,
                xk,
                grad_k,
                xv,
                grad_v,
                &self.grad_receptance_weight,
                &self.grad_key_weight,
                &self.grad_value_weight,
            ],
            bytemuck::bytes_of(linear_push),
            weight_groups,
        )
    }

    fn available_projection_weight_grad_topologies(&self) -> Vec<ProjectionWeightGradTopology> {
        let mut candidates = vec![ProjectionWeightGradTopology::Baseline];
        if self.linear3_weight_grad_tiled.is_some() {
            candidates.push(ProjectionWeightGradTopology::Tiled);
        }
        candidates
    }

    fn structural_projection_weight_grad_topology(&self) -> ProjectionWeightGradTopology {
        if self.linear3_weight_grad_tiled.is_some() {
            ProjectionWeightGradTopology::Tiled
        } else {
            ProjectionWeightGradTopology::Baseline
        }
    }

    fn choose_projection_weight_grad_topology(
        &self,
        rows: usize,
    ) -> Result<ProjectionWeightGradDecision> {
        let candidates = self.available_projection_weight_grad_topologies();
        let baseline = ProjectionWeightGradTopology::Baseline;
        if std::env::var_os(HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_DISABLE_AUTOTUNE_ENV)
            .is_some()
        {
            return Ok(ProjectionWeightGradDecision {
                topology: self.structural_projection_weight_grad_topology(),
                autotuned: false,
            });
        }
        if candidates.len() == 1 {
            return Ok(ProjectionWeightGradDecision {
                topology: baseline,
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let key = ProjectionWeightGradAutotuneKey {
            device_name: self.device.name().to_owned(),
            subgroup_size,
            width: self.width,
            rows,
        };
        let cache =
            PROJECTION_WEIGHT_GRAD_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(decision) = cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV projection weight-gradient autotune cache lock was poisoned")
            })?
            .get(&key)
            .copied()
        {
            return Ok(decision);
        }

        if let Err(err) = self.time_projection_weight_grad_topology_ms(rows, baseline) {
            if std::env::var_os(HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_AUTOTUNE_LOG_ENV)
                .is_some()
            {
                eprintln!(
                    "RWKV projection weight-gradient autotune warmup failed device={} rows={rows}: {err:#}; using baseline",
                    self.device.name()
                );
            }
            return Ok(ProjectionWeightGradDecision {
                topology: baseline,
                autotuned: false,
            });
        }

        let mut samples = vec![Vec::with_capacity(3); candidates.len()];
        for round in 0..3 {
            let indices: Box<dyn Iterator<Item = usize>> = if round % 2 == 0 {
                Box::new(0..candidates.len())
            } else {
                Box::new((0..candidates.len()).rev())
            };
            for index in indices {
                let topology = candidates[index];
                match self.time_projection_weight_grad_topology_ms(rows, topology) {
                    Ok(ms) => samples[index].push(ms),
                    Err(err) => {
                        if std::env::var_os(
                            HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_AUTOTUNE_LOG_ENV,
                        )
                        .is_some()
                        {
                            eprintln!(
                                "RWKV projection weight-gradient autotune failed device={} rows={rows} candidate={}: {err:#}; using baseline",
                                self.device.name(),
                                topology.label()
                            );
                        }
                        return Ok(ProjectionWeightGradDecision {
                            topology: baseline,
                            autotuned: false,
                        });
                    }
                }
            }
        }

        let mut timings = Vec::with_capacity(candidates.len());
        for (index, topology) in candidates.iter().copied().enumerate() {
            samples[index].sort_by(|lhs, rhs| lhs.total_cmp(rhs));
            timings.push((topology, samples[index][1]));
        }
        let selected = select_projection_traffic_topology(&timings, baseline);
        let decision = ProjectionWeightGradDecision {
            topology: selected,
            autotuned: true,
        };

        if std::env::var_os(HIERARCHOS_RWKV_PROJECTION_WEIGHT_GRAD_TILED_AUTOTUNE_LOG_ENV).is_some()
        {
            let summary = timings
                .iter()
                .map(|(topology, ms)| format!("{}={ms:.4}ms", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV projection weight-gradient autotune device={} subgroup={} width={} rows={} {} selected={} autotuned={}",
                self.device.name(),
                subgroup_size,
                self.width,
                rows,
                summary,
                selected.label(),
                decision.autotuned
            );
        }

        cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV projection weight-gradient autotune cache lock was poisoned")
            })?
            .insert(key, decision);
        Ok(decision)
    }

    fn time_projection_weight_grad_topology_ms(
        &self,
        batch: usize,
        topology: ProjectionWeightGradTopology,
    ) -> Result<f64> {
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let weight_groups = [
            div_ceil_u32(self.width, 16),
            div_ceil_u32(self.width, 16),
            1,
        ];
        let repetitions = if batch.saturating_mul(self.width) >= 4096 {
            16
        } else {
            64
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..repetitions {
            self.record_projection_weight_grad_topology_with_inputs(
                &mut commands,
                &self.xr,
                &self.xk,
                &self.xv,
                &self.grad_r,
                &self.grad_raw_k,
                &self.grad_v,
                &linear_push,
                weight_groups,
                topology,
            )?;
        }
        let started = Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_projection_time_mix_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_r: &GpuBuffer,
        grad_k: &GpuBuffer,
        grad_v: &GpuBuffer,
        mix_push: &MixPush,
        linear_push: &LinearPush,
        input_groups: [u32; 3],
        channel_groups: [u32; 3],
    ) -> Result<()> {
        self.record_projection_time_mix_backward_mode(
            commands,
            x_norm,
            previous,
            grad_r,
            grad_k,
            grad_v,
            mix_push,
            linear_push,
            input_groups,
            channel_groups,
            self.linear3_time_mix_backward_fused.is_some(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_projection_time_mix_backward_mode(
        &self,
        commands: &mut vulkan::ComputeBatch,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_r: &GpuBuffer,
        grad_k: &GpuBuffer,
        grad_v: &GpuBuffer,
        mix_push: &MixPush,
        linear_push: &LinearPush,
        input_groups: [u32; 3],
        channel_groups: [u32; 3],
        fused: bool,
    ) -> Result<()> {
        debug_assert!(self.backward_source_scale.is_finite());
        debug_assert!(self.backward_source_scale > 0.0);
        if fused && !self.native_fp16_projection_input_grad {
            let kernel = self
                .linear3_time_mix_backward_fused
                .as_ref()
                .context("RWKV fused projection/time-mix backward kernel is unavailable")?;
            return kernel.record_dispatch(
                commands,
                &[
                    x_norm,
                    previous,
                    &self.mix_r,
                    &self.mix_k,
                    &self.mix_v,
                    grad_r,
                    &self.receptance_weight,
                    grad_k,
                    &self.key_weight,
                    grad_v,
                    &self.value_weight,
                    &self.grad_x_norm,
                    &self.grad_previous,
                    &self.grad_mix_r,
                    &self.grad_mix_k,
                    &self.grad_mix_v,
                ],
                bytemuck::bytes_of(mix_push),
                channel_groups,
            );
        }

        if self.native_fp16_projection_input_grad {
            let input_grad_kernel = if self.source_scaled_backward_domain {
                self.linear3_input_grad_fp16_source_scaled
                    .as_ref()
                    .context(
                        "source-scaled native-FP16 RWKV projection input-gradient kernel is unavailable",
                    )?
            } else {
                self.linear3_input_grad_fp16_scaled
                    .as_ref()
                    .context("native-FP16 RWKV projection input-gradient kernel is unavailable")?
            };
            input_grad_kernel.record_dispatch(
                commands,
                &[
                    grad_r,
                    &self.receptance_weight,
                    grad_k,
                    &self.key_weight,
                    grad_v,
                    &self.value_weight,
                    &self.grad_xr,
                    &self.grad_xk,
                    &self.grad_xv,
                ],
                bytemuck::bytes_of(linear_push),
                input_groups,
            )?;
        } else {
            let decision = self.choose_projection_input_grad_topology(linear_push.rows as usize)?;
            self.record_projection_input_grad_topology(
                commands,
                grad_r,
                grad_k,
                grad_v,
                linear_push,
                input_groups,
                decision.topology,
            )?;
        }
        self.time_mix_backward.record_dispatch(
            commands,
            &[
                x_norm,
                previous,
                &self.mix_r,
                &self.mix_k,
                &self.mix_v,
                &self.grad_xr,
                &self.grad_xk,
                &self.grad_xv,
                &self.grad_x_norm,
                &self.grad_previous,
                &self.grad_mix_r,
                &self.grad_mix_k,
                &self.grad_mix_v,
            ],
            bytemuck::bytes_of(mix_push),
            channel_groups,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_projection_input_grad_topology(
        &self,
        commands: &mut vulkan::ComputeBatch,
        grad_r: &GpuBuffer,
        grad_k: &GpuBuffer,
        grad_v: &GpuBuffer,
        linear_push: &LinearPush,
        baseline_groups: [u32; 3],
        topology: ProjectionInputGradTopology,
    ) -> Result<()> {
        let (kernel, groups) = match topology {
            ProjectionInputGradTopology::Baseline => (&self.linear3_input_grad, baseline_groups),
            ProjectionInputGradTopology::WeightReuse { rows, tile } => {
                if linear_push.rows < rows {
                    bail!(
                        "RWKV projection input-gradient weight reuse{rows} requires at least {rows} rows"
                    );
                }
                let kernel = self
                    .linear3_input_grad_weight_reuse
                    .get(rows, tile)
                    .with_context(|| {
                        format!(
                            "RWKV projection input-gradient weight-reuse{rows} tile {tile} is unavailable"
                        )
                    })?;
                let groups = [
                    div_ceil_u32(linear_push.input_dim as usize, 64),
                    div_ceil_u32(linear_push.rows as usize, rows as usize),
                    1,
                ];
                (kernel, groups)
            }
        };
        kernel.record_dispatch(
            commands,
            &[
                grad_r,
                &self.receptance_weight,
                grad_k,
                &self.key_weight,
                grad_v,
                &self.value_weight,
                &self.grad_xr,
                &self.grad_xk,
                &self.grad_xv,
            ],
            bytemuck::bytes_of(linear_push),
            groups,
        )
    }

    fn available_projection_input_grad_topologies(
        &self,
        batch: usize,
    ) -> Vec<ProjectionInputGradTopology> {
        let mut candidates = vec![ProjectionInputGradTopology::Baseline];
        for rows in [2u32, 4, 8] {
            if batch < rows as usize {
                continue;
            }
            let Some(kernels) = self.linear3_input_grad_weight_reuse.rows(rows) else {
                continue;
            };
            candidates.extend(
                kernels
                    .available_tiles()
                    .into_iter()
                    .map(|tile| ProjectionInputGradTopology::WeightReuse { rows, tile }),
            );
        }
        candidates
    }

    fn structural_projection_input_grad_topology(
        &self,
        batch: usize,
    ) -> ProjectionInputGradTopology {
        for rows in [8u32, 4, 2] {
            if batch < rows as usize {
                continue;
            }
            if let Some(tile) = self
                .linear3_input_grad_weight_reuse
                .rows(rows)
                .and_then(WeightReuseKernels::largest_tile)
            {
                return ProjectionInputGradTopology::WeightReuse { rows, tile };
            }
        }
        ProjectionInputGradTopology::Baseline
    }

    fn choose_projection_input_grad_topology(
        &self,
        batch: usize,
    ) -> Result<ProjectionInputGradDecision> {
        let candidates = self.available_projection_input_grad_topologies(batch);
        let baseline = ProjectionInputGradTopology::Baseline;
        if std::env::var_os(HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_DISABLE_AUTOTUNE_ENV)
            .is_some()
        {
            return Ok(ProjectionInputGradDecision {
                topology: self.structural_projection_input_grad_topology(batch),
                autotuned: false,
            });
        }
        if candidates.len() == 1 {
            return Ok(ProjectionInputGradDecision {
                topology: baseline,
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let key = ProjectionInputGradAutotuneKey {
            device_name: self.device.name().to_owned(),
            subgroup_size,
            width: self.width,
            batch_pairs: batch.div_ceil(2),
            has_unpaired_tail: batch % 2 != 0,
        };
        let cache = PROJECTION_INPUT_GRAD_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(decision) = cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV projection input-gradient autotune cache lock was poisoned")
            })?
            .get(&key)
            .copied()
        {
            return Ok(decision);
        }

        if let Err(err) = self.time_projection_input_grad_topology_ms(batch, baseline) {
            if std::env::var_os(HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_AUTOTUNE_LOG_ENV)
                .is_some()
            {
                eprintln!(
                    "RWKV projection input-gradient autotune warmup failed device={} batch={batch}: {err:#}; using baseline",
                    self.device.name()
                );
            }
            return Ok(ProjectionInputGradDecision {
                topology: baseline,
                autotuned: false,
            });
        }

        let reuse_coordinates = candidates
            .iter()
            .filter_map(|topology| match *topology {
                ProjectionInputGradTopology::Baseline => None,
                ProjectionInputGradTopology::WeightReuse { rows, tile } => {
                    Some(ProjectionReuseCoordinate { rows, tile })
                }
            })
            .collect::<Vec<_>>();
        let preferred_reuse = match self.structural_projection_input_grad_topology(batch) {
            ProjectionInputGradTopology::Baseline => None,
            ProjectionInputGradTopology::WeightReuse { rows, tile } => {
                Some(ProjectionReuseCoordinate { rows, tile })
            }
        };
        let reuse_probe_coordinates =
            factorized_projection_reuse_probe_coordinates(&reuse_coordinates, preferred_reuse);
        let mut probe_candidates = Vec::with_capacity(reuse_probe_coordinates.len() + 1);
        probe_candidates.push(baseline);
        for coordinate in &reuse_probe_coordinates {
            let topology = ProjectionInputGradTopology::WeightReuse {
                rows: coordinate.rows,
                tile: coordinate.tile,
            };
            if candidates.contains(&topology) && !probe_candidates.contains(&topology) {
                probe_candidates.push(topology);
            }
        }
        let mut timings = match median_interleaved_timings(&probe_candidates, |topology| {
            self.time_projection_input_grad_topology_ms(batch, topology)
                .with_context(|| {
                    format!(
                        "timing RWKV projection input-gradient candidate {}",
                        topology.label()
                    )
                })
        }) {
            Ok(timings) => timings,
            Err(err) => {
                if std::env::var_os(HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_AUTOTUNE_LOG_ENV)
                    .is_some()
                {
                    eprintln!(
                        "RWKV projection input-gradient factorized autotune failed device={} batch={batch}: {err:#}; using baseline",
                        self.device.name()
                    );
                }
                return Ok(ProjectionInputGradDecision {
                    topology: baseline,
                    autotuned: false,
                });
            }
        };
        let reuse_timings = timings
            .iter()
            .filter_map(|(topology, ms)| match *topology {
                ProjectionInputGradTopology::Baseline => None,
                ProjectionInputGradTopology::WeightReuse { rows, tile } => {
                    Some((ProjectionReuseCoordinate { rows, tile }, *ms))
                }
            })
            .collect::<Vec<_>>();
        let synthesized_reuse = reuse_probe_coordinates.first().copied().and_then(|anchor| {
            select_factorized_projection_reuse_coordinate(
                &reuse_coordinates,
                &reuse_timings,
                anchor,
            )
        });
        let synthesized_was_unmeasured = synthesized_reuse.is_some_and(|coordinate| {
            !reuse_timings
                .iter()
                .any(|(measured, _)| *measured == coordinate)
        });
        if let Some(coordinate) = synthesized_reuse.filter(|coordinate| {
            !reuse_timings
                .iter()
                .any(|(measured, _)| measured == coordinate)
        }) {
            let topology = ProjectionInputGradTopology::WeightReuse {
                rows: coordinate.rows,
                tile: coordinate.tile,
            };
            match median_interleaved_timings(&[topology], |candidate| {
                self.time_projection_input_grad_topology_ms(batch, candidate)
                    .with_context(|| {
                        format!(
                            "timing synthesized RWKV projection input-gradient candidate {}",
                            candidate.label()
                        )
                    })
            }) {
                Ok(mut synthesized_timing) => timings.append(&mut synthesized_timing),
                Err(err) => {
                    if std::env::var_os(
                        HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_AUTOTUNE_LOG_ENV,
                    )
                    .is_some()
                    {
                        eprintln!(
                            "RWKV projection input-gradient synthesized factorized candidate failed device={} batch={batch} candidate={}: {err:#}; using baseline",
                            self.device.name(),
                            topology.label()
                        );
                    }
                    return Ok(ProjectionInputGradDecision {
                        topology: baseline,
                        autotuned: false,
                    });
                }
            }
        }
        let selected = select_projection_traffic_topology(&timings, baseline);
        let decision = ProjectionInputGradDecision {
            topology: selected,
            autotuned: true,
        };

        if std::env::var_os(HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_AUTOTUNE_LOG_ENV).is_some()
        {
            let summary = timings
                .iter()
                .map(|(topology, ms)| format!("{}={ms:.4}ms", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV projection input-gradient autotune device={} subgroup={} width={} batch_pairs={} tail={} policy=factorized-rows-tile measured_reuse_arms={} full_reuse_arms={} synthesized_unmeasured={} {} selected={} autotuned={}",
                self.device.name(),
                subgroup_size,
                self.width,
                key.batch_pairs,
                key.has_unpaired_tail,
                timings.len().saturating_sub(1),
                reuse_coordinates.len(),
                synthesized_was_unmeasured,
                summary,
                selected.label(),
                decision.autotuned
            );
        }

        cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV projection input-gradient autotune cache lock was poisoned")
            })?
            .insert(key, decision);
        Ok(decision)
    }

    fn time_projection_input_grad_topology_ms(
        &self,
        batch: usize,
        topology: ProjectionInputGradTopology,
    ) -> Result<f64> {
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let baseline_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let repetitions = if batch.saturating_mul(self.width) >= 4096 {
            4
        } else {
            16
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..repetitions {
            self.record_projection_input_grad_topology(
                &mut commands,
                &self.grad_r,
                &self.grad_raw_k,
                &self.grad_v,
                &linear_push,
                baseline_groups,
                topology,
            )?;
        }
        let started = Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_projection_backward_schedule(
        &self,
        commands: &mut vulkan::ComputeBatch,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_r: &GpuBuffer,
        grad_k: &GpuBuffer,
        grad_v: &GpuBuffer,
        mix_push: &MixPush,
        linear_push: &LinearPush,
        input_groups: [u32; 3],
        weight_groups: [u32; 3],
        channel_groups: [u32; 3],
        schedule: ProjectionBackwardSchedule,
    ) -> Result<()> {
        let record_weight = |commands: &mut vulkan::ComputeBatch| {
            self.record_projection_weight_grad(
                commands,
                grad_r,
                grad_k,
                grad_v,
                linear_push,
                weight_groups,
            )
        };
        let record_input_mix = |commands: &mut vulkan::ComputeBatch| {
            self.record_projection_time_mix_backward_mode(
                commands,
                x_norm,
                previous,
                grad_r,
                grad_k,
                grad_v,
                mix_push,
                linear_push,
                input_groups,
                channel_groups,
                schedule.uses_fused_mix(),
            )
        };

        if schedule.weight_first() {
            record_weight(commands)?;
            record_input_mix(commands)
        } else {
            record_input_mix(commands)?;
            record_weight(commands)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_key_state_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        key_push: &KeyPush,
        state_push: &StatePush,
        head_groups: [u32; 3],
        vector_groups: [u32; 3],
    ) -> Result<()> {
        if self.head_size <= 64 {
            let groups = u32::try_from(
                batch
                    .checked_mul(self.heads)
                    .context("RWKV fused key/state-forward workgroup count overflow")?,
            )
            .context("RWKV fused key/state-forward workgroup count exceeds Vulkan u32 range")?;
            return self.key_state_forward_fused.record_dispatch(
                commands,
                &[
                    &self.raw_k,
                    a,
                    &self.k_k,
                    &self.k_a,
                    state,
                    &self.r,
                    &self.v,
                    w,
                    &self.scaled_k,
                    &self.kk,
                    &self.new_state,
                    &self.tmix,
                    &self.saved_sa,
                ],
                bytemuck::bytes_of(state_push),
                [groups, 1, 1],
            );
        }

        self.key_transform_forward.record_dispatch(
            commands,
            &[
                &self.raw_k,
                a,
                &self.k_k,
                &self.k_a,
                &self.scaled_k,
                &self.kk,
            ],
            bytemuck::bytes_of(key_push),
            head_groups,
        )?;
        self.state_forward.record_dispatch(
            commands,
            &[
                state,
                &self.r,
                &self.scaled_k,
                &self.v,
                &self.kk,
                a,
                w,
                &self.new_state,
                &self.tmix,
                &self.saved_sa,
            ],
            bytemuck::bytes_of(state_push),
            vector_groups,
        )
    }

    fn configure_state_backward_schedules(&mut self) -> Result<()> {
        if self.head_size > 64 || self.state_backward_fused_rkv_add3.is_none() {
            return Ok(());
        }

        self.state_backward_schedule_batch1 = Some(self.choose_state_backward_schedule(1)?);
        if self.max_batch > 1 {
            self.state_backward_profile_batch = self.max_batch.min(8);
            self.state_backward_schedule_multi =
                Some(self.choose_state_backward_schedule(self.state_backward_profile_batch)?);
        }
        Ok(())
    }

    fn configure_backward_segment_schedules(&mut self, full_cell: bool) -> Result<()> {
        if self.head_size > 64 || self.state_backward_fused_rkv_add3.is_none() {
            return Ok(());
        }

        self.backward_segment_schedule_batch1 =
            Some(self.choose_backward_segment_schedule(1, full_cell)?);
        if self.max_batch > 1 {
            self.backward_segment_schedule_multi =
                Some(self.choose_backward_segment_schedule(
                    self.state_backward_profile_batch,
                    full_cell,
                )?);
        }
        Ok(())
    }

    fn available_projection_backward_schedules(&self) -> Vec<ProjectionBackwardSchedule> {
        let mut schedules = vec![
            ProjectionBackwardSchedule::SplitMixThenWeight,
            ProjectionBackwardSchedule::WeightThenSplitMix,
        ];
        if self.linear3_time_mix_backward_fused.is_some() {
            schedules.push(ProjectionBackwardSchedule::FusedMixThenWeight);
            schedules.push(ProjectionBackwardSchedule::WeightThenFusedMix);
        }
        schedules
    }

    fn available_backward_segment_schedules(
        &self,
        batch: usize,
        full_cell: bool,
    ) -> Result<Vec<BackwardSegmentSchedule>> {
        let forced_state = forced_state_backward_schedule()?;
        let forced_projection = forced_projection_backward_schedule()?;
        let forced_low_rank_fan_in = if full_cell {
            forced_low_rank_fan_in_schedule()?
        } else {
            None
        };
        let state_schedules = self.available_state_backward_schedules(batch);
        let projection_schedules = self.available_projection_backward_schedules();
        let low_rank_fan_in_schedules = if full_cell {
            self.low_rank
                .as_ref()
                .context("RWKV full-cell backward schedule search requires low-rank a/w/g")?
                .available_backward_fan_in_schedules(true)
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>()
        } else {
            vec![None]
        };
        let mut schedules = Vec::with_capacity(
            state_schedules.len() * projection_schedules.len() * low_rank_fan_in_schedules.len(),
        );

        for state in state_schedules {
            if forced_state.is_some_and(|forced| forced != state) {
                continue;
            }
            for &projection in &projection_schedules {
                if forced_projection.is_some_and(|forced| forced != projection) {
                    continue;
                }
                for &low_rank_fan_in in &low_rank_fan_in_schedules {
                    if forced_low_rank_fan_in.is_some_and(|forced| Some(forced) != low_rank_fan_in)
                    {
                        continue;
                    }
                    schedules.push(BackwardSegmentSchedule {
                        state,
                        projection,
                        low_rank_fan_in,
                    });
                }
            }
        }
        Ok(schedules)
    }

    fn choose_backward_segment_schedule(
        &self,
        batch: usize,
        full_cell: bool,
    ) -> Result<BackwardSegmentDecision> {
        let candidates = self.available_backward_segment_schedules(batch, full_cell)?;
        let default = candidates.last().copied().context(
            "RWKV backward segment has no compatible recurrence/projection/low-rank schedule",
        )?;

        if candidates.len() == 1
            || std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_DISABLE_AUTOTUNE").is_some()
        {
            return Ok(BackwardSegmentDecision {
                schedule: default,
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let (w_rank, a_rank, g_rank) = if full_cell {
            self.low_rank
                .as_ref()
                .context("RWKV full-cell backward autotune requires low-rank a/w/g")?
                .ranks()
        } else {
            (0, 0, 0)
        };
        let key = BackwardSegmentAutotuneKey {
            device_name: self.device.name().to_owned(),
            subgroup_size,
            width: self.width,
            head_size: self.head_size,
            batch,
            w_rank,
            a_rank,
            g_rank,
            full_cell,
        };
        let cache = BACKWARD_SEGMENT_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(decision) = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("RWKV backward-segment autotune cache lock was poisoned"))?
            .get(&key)
            .copied()
        {
            return Ok(decision);
        }

        if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_REAUTOTUNE").is_none() {
            match load_persistent_backward_segment_schedule(&key, &candidates) {
                Ok(Some(schedule)) => {
                    let decision = BackwardSegmentDecision {
                        schedule,
                        autotuned: true,
                    };
                    cache
                        .lock()
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "RWKV backward-segment autotune cache lock was poisoned"
                            )
                        })?
                        .insert(key, decision);
                    if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG").is_some() {
                        eprintln!(
                            "RWKV backward-segment autotune persistent-hit device={} subgroup={} width={} head_size={} batch={} ranks={}/{}/{} scope={} selected={}",
                            self.device.name(),
                            subgroup_size,
                            self.width,
                            self.head_size,
                            batch,
                            w_rank,
                            a_rank,
                            g_rank,
                            if full_cell { "full-cell" } else { "core" },
                            schedule.label()
                        );
                    }
                    return Ok(decision);
                }
                Ok(None) => {}
                Err(err) => {
                    if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG").is_some() {
                        eprintln!(
                            "RWKV backward-segment persistent autotune cache read failed: {err:#}"
                        );
                    }
                }
            }
        }

        let timings = match self.measure_backward_segment_schedules(
            batch,
            &candidates,
            default,
            full_cell,
        ) {
            Ok(timings) => timings,
            Err(err) => {
                if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG").is_some() {
                    eprintln!(
                        "RWKV backward-segment autotune failed device={} width={} head_size={} batch={} scope={}: {err:#}; using deepest compatible schedule",
                        self.device.name(),
                        self.width,
                        self.head_size,
                        batch,
                        if full_cell { "full-cell" } else { "core" }
                    );
                }
                return Ok(BackwardSegmentDecision {
                    schedule: default,
                    autotuned: false,
                });
            }
        };

        let selected = select_backward_segment_schedule(&timings, default);
        let decision = BackwardSegmentDecision {
            schedule: selected,
            autotuned: true,
        };

        if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG").is_some() {
            let summary = timings
                .iter()
                .map(|(schedule, ms)| format!("{}={ms:.4}ms", schedule.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV backward-segment autotune device={} subgroup={} width={} head_size={} batch={} scope={} {} selected={}",
                self.device.name(),
                subgroup_size,
                self.width,
                self.head_size,
                batch,
                if full_cell { "full-cell" } else { "core" },
                summary,
                selected.label()
            );
        }

        cache
            .lock()
            .map_err(|_| anyhow::anyhow!("RWKV backward-segment autotune cache lock was poisoned"))?
            .insert(key.clone(), decision);
        if let Err(err) = store_persistent_backward_segment_schedule(&key, selected) {
            if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG").is_some() {
                eprintln!("RWKV backward-segment persistent autotune cache write failed: {err:#}");
            }
        }
        Ok(decision)
    }

    fn measure_backward_segment_schedules(
        &self,
        batch: usize,
        candidates: &[BackwardSegmentSchedule],
        default: BackwardSegmentSchedule,
        full_cell: bool,
    ) -> Result<Vec<(BackwardSegmentSchedule, f64)>> {
        // Bring an idle/low-clock GPU into a steady regime using one fixed
        // topology. This avoids baking DVFS ramp order into the schedule choice.
        let mut previous_ms: Option<f64> = None;
        let mut stable_samples = 0usize;
        for _ in 0..24 {
            let ms = self.time_profiled_backward_segment_schedule_ms(batch, default, full_cell)?;
            if let Some(previous) = previous_ms {
                let relative_delta = (ms - previous).abs() / previous.max(1.0e-9);
                if relative_delta <= 0.03 {
                    stable_samples += 1;
                    if stable_samples >= 3 {
                        break;
                    }
                } else {
                    stable_samples = 0;
                }
            }
            previous_ms = Some(ms);
        }

        let finalists = if candidates.len() > BACKWARD_SEGMENT_ELIMINATION_THRESHOLD {
            // The full-cell search is a 3-D Cartesian product. One cheap pass
            // estimates the useful half of each axis before the robust race.
            // The overall one-shot winner and the default are always retained
            // so a noisy marginal estimate cannot eliminate both anchors.
            let mut elimination = Vec::with_capacity(candidates.len());
            for &schedule in candidates {
                elimination.push((
                    schedule,
                    self.time_profiled_backward_segment_schedule_ms(batch, schedule, full_cell)?,
                ));
            }
            let finalists = prune_backward_segment_candidates(&elimination, default);
            if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG").is_some() {
                let labels = finalists
                    .iter()
                    .map(|schedule| schedule.label())
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "RWKV backward-segment autotune elimination candidates={} finalists={} [{}]",
                    candidates.len(),
                    finalists.len(),
                    labels
                );
            }
            finalists
        } else {
            candidates.to_vec()
        };

        // Interleave finalist samples and reverse traversal every other round.
        // Any residual clock/thermal drift is therefore distributed across the
        // reduced set instead of favoring the schedules measured last.
        let mut samples = vec![Vec::with_capacity(5); finalists.len()];
        for round in 0..5 {
            if round % 2 == 0 {
                for (index, &schedule) in finalists.iter().enumerate() {
                    samples[index].push(
                        self.time_profiled_backward_segment_schedule_ms(
                            batch, schedule, full_cell,
                        )?,
                    );
                }
            } else {
                for (index, &schedule) in finalists.iter().enumerate().rev() {
                    samples[index].push(
                        self.time_profiled_backward_segment_schedule_ms(
                            batch, schedule, full_cell,
                        )?,
                    );
                }
            }
        }

        let mut timings = Vec::with_capacity(finalists.len());
        for (index, &schedule) in finalists.iter().enumerate() {
            samples[index].sort_by(|lhs, rhs| lhs.total_cmp(rhs));
            timings.push((schedule, samples[index][2]));
        }
        Ok(timings)
    }

    fn time_profiled_backward_segment_schedule_ms(
        &self,
        batch: usize,
        schedule: BackwardSegmentSchedule,
        full_cell: bool,
    ) -> Result<f64> {
        if full_cell {
            self.time_full_cell_backward_segment_schedule_ms(batch, schedule)
        } else {
            self.time_backward_segment_schedule_ms(batch, schedule)
        }
    }

    #[cfg(test)]
    fn median_backward_segment_schedule_ms(
        &self,
        batch: usize,
        schedule: BackwardSegmentSchedule,
    ) -> Result<f64> {
        let _ = self.time_backward_segment_schedule_ms(batch, schedule)?;
        let mut samples = [0.0f64; 5];
        for sample in &mut samples {
            *sample = self.time_backward_segment_schedule_ms(batch, schedule)?;
        }
        samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
        Ok(samples[2])
    }

    fn time_backward_segment_schedule_ms(
        &self,
        batch: usize,
        schedule: BackwardSegmentSchedule,
    ) -> Result<f64> {
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV backward-segment vector size overflow")?;
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let input_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let weight_groups = [
            div_ceil_u32(self.width, 16),
            div_ceil_u32(self.width, 16),
            1,
        ];
        let channel_groups = [div_ceil_u32(self.width, 64), 1, 1];
        // These segments are short enough that single-submit host/driver noise
        // can rival the differences between schedules. Amortize that noise by
        // recording several complete recurrence+projection segments per sample;
        // large batches already provide enough GPU work to need fewer repeats.
        let repetitions = if vector_len >= 4096 { 4 } else { 32 };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..repetitions {
            self.record_state_backward_schedule(
                &mut commands,
                batch,
                &self.state,
                &self.a,
                &self.w,
                &self.grad_new_state,
                &self.grad_tmix,
                &self.grad_r,
                &self.grad_scaled_k,
                &self.grad_v,
                &self.grad_g_external,
                &state_push,
                schedule.state,
            )?;
            self.record_projection_backward_schedule(
                &mut commands,
                &self.x_norm,
                &self.previous,
                &self.total_grad_r,
                &self.grad_raw_k,
                &self.total_grad_v_external,
                &mix_push,
                &linear_push,
                input_groups,
                weight_groups,
                channel_groups,
                schedule.projection,
            )?;
        }
        let started = Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    fn time_full_cell_backward_segment_schedule_ms(
        &self,
        batch: usize,
        schedule: BackwardSegmentSchedule,
    ) -> Result<f64> {
        let low_rank = self
            .low_rank
            .as_ref()
            .context("RWKV full-cell backward profiling requires the low-rank a/w/g graph")?;
        let post_mix = self
            .post_mix
            .as_ref()
            .context("RWKV full-cell backward profiling requires the post-mix graph")?;
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV full-cell backward profile vector size overflow")?;
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let input_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let weight_groups = [
            div_ceil_u32(self.width, 16),
            div_ceil_u32(self.width, 16),
            1,
        ];
        let channel_groups = [div_ceil_u32(self.width, 64), 1, 1];
        let len_push = LenPush {
            len: vector_len as u32,
        };
        let add_groups = [div_ceil_u32(vector_len, 256), 1, 1];

        // The complete cell slice contains enough dispatches to amortize host
        // submission overhead with far fewer repeats than the core-only
        // recurrence/projection microsegment. Repeating the whole slice in one
        // command buffer still exposes descriptor reuse, cache pressure, and
        // RAW dependencies between the candidate projection topology and the
        // low-rank/shared-input tail. The tail is part of the schedule now, so
        // the profiler races split, base-fused, and (where descriptor limits
        // permit) outer-input-fused fan-in against recurrence/projection.
        let repetitions = if vector_len >= 4096 { 2 } else { 4 };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let low_rank_fan_in = schedule
            .low_rank_fan_in
            .context("RWKV full-cell backward schedule is missing low-rank fan-in topology")?;
        for _ in 0..repetitions {
            post_mix.record_backward(
                &mut commands,
                batch,
                &self.tmix,
                &self.r,
                &self.scaled_k,
                &self.v,
                low_rank.g_buffer(),
                &self.grad_g_external,
            )?;

            self.record_backward_segment_schedule(
                &mut commands,
                batch,
                &self.state,
                low_rank.a_buffer(),
                low_rank.w_buffer(),
                &self.grad_new_state,
                post_mix.grad_tmix_buffer(),
                post_mix.grad_r_buffer(),
                post_mix.grad_k_buffer(),
                post_mix.grad_v_buffer(),
                &self.grad_g_external,
                &self.x_norm,
                &self.previous,
                &state_push,
                &mix_push,
                &linear_push,
                input_groups,
                weight_groups,
                channel_groups,
                schedule,
            )?;

            // `grad_g_external` is a resident vector-sized buffer and serves as
            // a representative enclosing-cell adjoint for profiling. Values do
            // not affect topology timing; keeping it resident captures the RAW
            // and descriptor pressure of the real full-cell path.
            let output_grad_x_norm = if low_rank_fan_in == RwkvLowRankFanInSchedule::FusedOuter {
                &self.total_grad_v
            } else {
                &self.fused_grad_x_norm
            };
            let (low_rank_accumulated, external_x_accumulated) = low_rank
                .record_backward_with_fan_in_schedule_and_workgroup_size(
                    &mut commands,
                    batch,
                    &self.x_norm,
                    &self.previous,
                    &self.grad_a,
                    &self.grad_w,
                    post_mix.grad_g_buffer(),
                    &self.grad_x_norm,
                    &self.grad_previous,
                    Some(&self.grad_g_external),
                    output_grad_x_norm,
                    &self.fused_grad_previous,
                    low_rank_fan_in,
                    self.selected_backward_kernel_geometry(batch)
                        .workgroup_size() as usize,
                )?;
            if !low_rank_accumulated {
                self.vector_add.record_dispatch(
                    &mut commands,
                    &[
                        &self.grad_x_norm,
                        low_rank.grad_x_norm_buffer(),
                        &self.fused_grad_x_norm,
                    ],
                    bytemuck::bytes_of(&len_push),
                    add_groups,
                )?;
                self.vector_add.record_dispatch(
                    &mut commands,
                    &[
                        &self.grad_previous,
                        low_rank.grad_previous_buffer(),
                        &self.fused_grad_previous,
                    ],
                    bytemuck::bytes_of(&len_push),
                    add_groups,
                )?;
            }
            if !external_x_accumulated {
                self.vector_add.record_dispatch(
                    &mut commands,
                    &[
                        &self.fused_grad_x_norm,
                        &self.grad_g_external,
                        &self.total_grad_v,
                    ],
                    bytemuck::bytes_of(&len_push),
                    add_groups,
                )?;
            }
        }
        let started = Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_backward_segment_schedule(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_tmix: &GpuBuffer,
        post_grad_r: &GpuBuffer,
        post_grad_k: &GpuBuffer,
        post_grad_v: &GpuBuffer,
        external_grad_v: &GpuBuffer,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        state_push: &StatePush,
        mix_push: &MixPush,
        linear_push: &LinearPush,
        input_groups: [u32; 3],
        weight_groups: [u32; 3],
        channel_groups: [u32; 3],
        schedule: BackwardSegmentSchedule,
    ) -> Result<()> {
        self.record_state_backward_schedule(
            commands,
            batch,
            state,
            a,
            w,
            grad_new_state,
            grad_tmix,
            post_grad_r,
            post_grad_k,
            post_grad_v,
            external_grad_v,
            state_push,
            schedule.state,
        )?;
        self.record_projection_backward_schedule(
            commands,
            x_norm,
            previous,
            &self.total_grad_r,
            &self.grad_raw_k,
            &self.total_grad_v_external,
            mix_push,
            linear_push,
            input_groups,
            weight_groups,
            channel_groups,
            schedule.projection,
        )
    }

    fn selected_backward_kernel_geometry(&self, batch: usize) -> BackwardKernelGeometry {
        if batch == 1 {
            self.backward_kernel_geometry_batch1
        } else {
            self.backward_kernel_geometry_multi
        }
    }

    fn recurrent_parallel_geometry_changes_order(&self, geometry: BackwardKernelGeometry) -> bool {
        let workgroup_size = geometry.workgroup_size() as usize;
        self.head_size > 0
            && workgroup_size > self.head_size
            && workgroup_size.is_multiple_of(self.head_size)
    }

    fn recurrent_subgroup_geometry_changes_order(&self, geometry: BackwardKernelGeometry) -> bool {
        let caps = self.device.subgroup_capabilities();
        caps.compute_supported
            && caps.basic_supported
            && caps.arithmetic_supported
            && recurrent_subgroup_geometry_supported(
                geometry.workgroup_size(),
                caps.subgroup_size,
                self.head_size,
            )
    }

    fn state_backward_kernel_for_geometry(
        &self,
        schedule: StateBackwardSchedule,
        geometry: BackwardKernelGeometry,
    ) -> Option<&vulkan::ComputeKernel> {
        self.state_backward_kernel_for_geometry_with_policy(
            schedule,
            geometry,
            self.numerics_policy,
        )
    }

    fn state_backward_kernel_for_geometry_with_policy(
        &self,
        schedule: StateBackwardSchedule,
        geometry: BackwardKernelGeometry,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Option<&vulkan::ComputeKernel> {
        if numerics_policy == RwkvNumericsPolicy::FastRecurrentSubgroup {
            if !self.recurrent_subgroup_geometry_changes_order(geometry) {
                return None;
            }
            return match (schedule, geometry) {
                (StateBackwardSchedule::RkvAdd3, BackwardKernelGeometry::Wg32) => self
                    .state_backward_fused_rkv_add3_recurrent_subgroup_wg32
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3, BackwardKernelGeometry::Wg64) => self
                    .state_backward_fused_rkv_add3_recurrent_subgroup
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3, BackwardKernelGeometry::Wg128) => self
                    .state_backward_fused_rkv_add3_recurrent_subgroup_wg128
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg32) => self
                    .state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg32
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg64) => self
                    .state_backward_fused_rkv_add3_key_transform_recurrent_subgroup
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg128) => self
                    .state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg128
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg32,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg32
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg64,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg128,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg128
                    .as_ref(),
            };
        }
        if numerics_policy == RwkvNumericsPolicy::FastRecurrentTree {
            if !self.recurrent_parallel_geometry_changes_order(geometry) {
                return match schedule {
                    StateBackwardSchedule::RkvAdd3 => match geometry {
                        BackwardKernelGeometry::Wg32 => {
                            self.state_backward_fused_rkv_add3_wg32.as_ref()
                        }
                        BackwardKernelGeometry::Wg64 => self.state_backward_fused_rkv_add3.as_ref(),
                        BackwardKernelGeometry::Wg128 => {
                            self.state_backward_fused_rkv_add3_wg128.as_ref()
                        }
                    },
                    StateBackwardSchedule::RkvAdd3KeyTransform
                    | StateBackwardSchedule::RkvAdd3KeyTransformReduce => None,
                };
            }
            let tree_kernel = match (schedule, geometry) {
                (StateBackwardSchedule::RkvAdd3, _) => None,
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg32) => self
                    .state_backward_fused_rkv_add3_key_transform_tree_wg32
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg64) => self
                    .state_backward_fused_rkv_add3_key_transform_tree
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg128) => self
                    .state_backward_fused_rkv_add3_key_transform_tree_wg128
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg32,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_tree_wg32
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg64,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_tree
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg128,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_tree_wg128
                    .as_ref(),
            };
            if tree_kernel.is_some() {
                return tree_kernel;
            }
        }
        if numerics_policy == RwkvNumericsPolicy::FastRecurrentTiled {
            if !self.recurrent_parallel_geometry_changes_order(geometry) {
                return match schedule {
                    StateBackwardSchedule::RkvAdd3 => match geometry {
                        BackwardKernelGeometry::Wg32 => {
                            self.state_backward_fused_rkv_add3_wg32.as_ref()
                        }
                        BackwardKernelGeometry::Wg64 => self.state_backward_fused_rkv_add3.as_ref(),
                        BackwardKernelGeometry::Wg128 => {
                            self.state_backward_fused_rkv_add3_wg128.as_ref()
                        }
                    },
                    StateBackwardSchedule::RkvAdd3KeyTransform
                    | StateBackwardSchedule::RkvAdd3KeyTransformReduce => None,
                };
            }
            let tiled_kernel = match (schedule, geometry) {
                (StateBackwardSchedule::RkvAdd3, _) => None,
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg32) => self
                    .state_backward_fused_rkv_add3_key_transform_tiled_wg32
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg64) => self
                    .state_backward_fused_rkv_add3_key_transform_tiled
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg128) => self
                    .state_backward_fused_rkv_add3_key_transform_tiled_wg128
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg32,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg32
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg64,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_tiled
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg128,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_tiled_wg128
                    .as_ref(),
            };
            if tiled_kernel.is_some() {
                return tiled_kernel;
            }
        }
        if numerics_policy == RwkvNumericsPolicy::FastSubgroup {
            let subgroup_kernel = match (schedule, geometry) {
                (StateBackwardSchedule::RkvAdd3, _) => None,
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg32) => self
                    .state_backward_fused_rkv_add3_key_transform_subgroup_wg32
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg64) => self
                    .state_backward_fused_rkv_add3_key_transform_subgroup
                    .as_ref(),
                (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg128) => self
                    .state_backward_fused_rkv_add3_key_transform_subgroup_wg128
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg32,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg32
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg64,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_subgroup
                    .as_ref(),
                (
                    StateBackwardSchedule::RkvAdd3KeyTransformReduce,
                    BackwardKernelGeometry::Wg128,
                ) => self
                    .state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg128
                    .as_ref(),
            };
            if subgroup_kernel.is_some() {
                return subgroup_kernel;
            }
        }

        match (schedule, geometry) {
            (StateBackwardSchedule::RkvAdd3, BackwardKernelGeometry::Wg32) => {
                self.state_backward_fused_rkv_add3_wg32.as_ref()
            }
            (StateBackwardSchedule::RkvAdd3, BackwardKernelGeometry::Wg64) => {
                self.state_backward_fused_rkv_add3.as_ref()
            }
            (StateBackwardSchedule::RkvAdd3, BackwardKernelGeometry::Wg128) => {
                self.state_backward_fused_rkv_add3_wg128.as_ref()
            }
            (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg32) => self
                .state_backward_fused_rkv_add3_key_transform_wg32
                .as_ref(),
            (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg64) => {
                self.state_backward_fused_rkv_add3_key_transform.as_ref()
            }
            (StateBackwardSchedule::RkvAdd3KeyTransform, BackwardKernelGeometry::Wg128) => self
                .state_backward_fused_rkv_add3_key_transform_wg128
                .as_ref(),
            (StateBackwardSchedule::RkvAdd3KeyTransformReduce, BackwardKernelGeometry::Wg32) => {
                self.state_backward_fused_rkv_add3_key_transform_reduce_wg32
                    .as_ref()
            }
            (StateBackwardSchedule::RkvAdd3KeyTransformReduce, BackwardKernelGeometry::Wg64) => {
                self.state_backward_fused_rkv_add3_key_transform_reduce
                    .as_ref()
            }
            (StateBackwardSchedule::RkvAdd3KeyTransformReduce, BackwardKernelGeometry::Wg128) => {
                self.state_backward_fused_rkv_add3_key_transform_reduce_wg128
                    .as_ref()
            }
        }
    }

    fn available_state_backward_schedules(&self, batch: usize) -> Vec<StateBackwardSchedule> {
        let mut schedules = Vec::with_capacity(3);
        if self.state_backward_fused_rkv_add3.is_some() {
            schedules.push(StateBackwardSchedule::RkvAdd3);
        }
        if self.state_backward_fused_rkv_add3_key_transform.is_some() {
            schedules.push(StateBackwardSchedule::RkvAdd3KeyTransform);
        }
        if batch == 1
            && self
                .state_backward_fused_rkv_add3_key_transform_reduce
                .is_some()
        {
            schedules.push(StateBackwardSchedule::RkvAdd3KeyTransformReduce);
        }
        schedules
    }

    fn record_key_transform_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        a: &GpuBuffer,
        grad_scaled_k: &GpuBuffer,
    ) -> Result<()> {
        let key_push = KeyPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let (kernel, groups) = if self.numerics_policy == RwkvNumericsPolicy::FastSubgroup {
            (
                self.key_transform_backward_subgroup.as_ref().context(
                    "RWKV fast-subgroup key normalization is unavailable on this Vulkan device",
                )?,
                [div_ceil_u32(batch * self.heads, 1), 1, 1],
            )
        } else {
            (
                &self.key_transform_backward,
                [div_ceil_u32(batch * self.heads, 64), 1, 1],
            )
        };
        kernel.record_dispatch(
            commands,
            &[
                &self.raw_k,
                a,
                &self.k_k,
                &self.k_a,
                &self.kk,
                grad_scaled_k,
                &self.grad_kk,
                &self.grad_a_direct,
                &self.grad_raw_k,
                &self.grad_a,
                &self.grad_k_k_partial,
                &self.grad_k_a_partial,
            ],
            bytemuck::bytes_of(&key_push),
            groups,
        )
    }

    fn available_backward_kernel_geometries(&self, batch: usize) -> Vec<BackwardKernelGeometry> {
        self.available_backward_kernel_geometries_for_numerics(batch, self.numerics_policy)
    }

    fn available_backward_kernel_geometries_for_numerics(
        &self,
        batch: usize,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Vec<BackwardKernelGeometry> {
        if self.head_size > 64 {
            return Vec::new();
        }
        let schedules = self.available_state_backward_schedules(batch);
        if schedules.is_empty() {
            return Vec::new();
        }
        let mut geometries = BackwardKernelGeometry::ALL
            .into_iter()
            .filter(|&geometry| {
                self.device
                    .supports_compute_work_group_size_x(geometry.workgroup_size())
                    && schedules.iter().all(|&schedule| {
                        self.state_backward_kernel_for_geometry_with_policy(
                            schedule,
                            geometry,
                            numerics_policy,
                        )
                        .is_some()
                    })
            })
            .collect::<Vec<_>>();
        let subgroup_size = self.device.subgroup_capabilities().subgroup_size.max(1);
        geometries
            .sort_by_key(|&geometry| backward_kernel_geometry_priority(geometry, subgroup_size));
        geometries
    }

    fn choose_state_backward_schedule(&self, batch: usize) -> Result<StateBackwardDecision> {
        let candidates = self.available_state_backward_schedules(batch);
        let default = candidates
            .last()
            .copied()
            .context("RWKV external-V state backward has no compatible fused schedule")?;

        if let Some(forced) = forced_state_backward_schedule()? {
            if !candidates.contains(&forced) {
                bail!(
                    "requested RWKV state-backward schedule {} is unavailable for batch={batch}, head_size={}, or this Vulkan descriptor limit",
                    forced.label(),
                    self.head_size
                );
            }
            return Ok(StateBackwardDecision {
                schedule: forced,
                autotuned: false,
            });
        }

        if candidates.len() == 1
            || std::env::var_os("HIERARCHOS_RWKV_STATE_BACKWARD_DISABLE_AUTOTUNE").is_some()
        {
            return Ok(StateBackwardDecision {
                schedule: default,
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let key = StateBackwardAutotuneKey {
            device_name: self.device.name().to_owned(),
            subgroup_size,
            width: self.width,
            head_size: self.head_size,
            batch,
        };
        let cache = STATE_BACKWARD_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(decision) = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("RWKV state-backward autotune cache lock was poisoned"))?
            .get(&key)
            .copied()
        {
            return Ok(decision);
        }

        let mut timings = Vec::with_capacity(candidates.len());
        for schedule in candidates {
            let ms = match self.median_state_backward_schedule_ms(batch, schedule) {
                Ok(ms) => ms,
                Err(err) => {
                    if std::env::var_os("HIERARCHOS_RWKV_STATE_BACKWARD_AUTOTUNE_LOG").is_some() {
                        eprintln!(
                            "RWKV state-backward autotune failed device={} width={} head_size={} batch={} candidate={}: {err:#}; using deepest compatible fusion",
                            self.device.name(),
                            self.width,
                            self.head_size,
                            batch,
                            schedule.label()
                        );
                    }
                    return Ok(StateBackwardDecision {
                        schedule: default,
                        autotuned: false,
                    });
                }
            };
            timings.push((schedule, ms));
        }
        let selected = select_state_backward_schedule(&timings, default);
        let decision = StateBackwardDecision {
            schedule: selected,
            autotuned: true,
        };

        if std::env::var_os("HIERARCHOS_RWKV_STATE_BACKWARD_AUTOTUNE_LOG").is_some() {
            let summary = timings
                .iter()
                .map(|(schedule, ms)| format!("{}={ms:.4}ms", schedule.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV state-backward autotune device={} subgroup={} width={} head_size={} batch={} {} selected={}",
                self.device.name(),
                subgroup_size,
                self.width,
                self.head_size,
                batch,
                summary,
                selected.label()
            );
        }

        cache
            .lock()
            .map_err(|_| anyhow::anyhow!("RWKV state-backward autotune cache lock was poisoned"))?
            .insert(key, decision);
        Ok(decision)
    }

    fn median_state_backward_schedule_ms(
        &self,
        batch: usize,
        schedule: StateBackwardSchedule,
    ) -> Result<f64> {
        let _ = self.time_state_backward_schedule_ms(batch, schedule)?;
        let mut samples = [0.0f64; 3];
        for sample in &mut samples {
            *sample = self.time_state_backward_schedule_ms(batch, schedule)?;
        }
        samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
        Ok(samples[1])
    }

    fn time_state_backward_schedule_ms(
        &self,
        batch: usize,
        schedule: StateBackwardSchedule,
    ) -> Result<f64> {
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let repetitions = if batch.saturating_mul(self.heads) > 128 {
            2
        } else {
            4
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..repetitions {
            self.record_state_backward_schedule(
                &mut commands,
                batch,
                &self.state,
                &self.a,
                &self.w,
                &self.grad_new_state,
                &self.grad_tmix,
                &self.grad_xr,
                &self.grad_xk,
                &self.grad_xv,
                &self.grad_g_external,
                &state_push,
                schedule,
            )?;
        }
        let started = Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_state_backward_schedule(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_tmix: &GpuBuffer,
        post_grad_r: &GpuBuffer,
        post_grad_k: &GpuBuffer,
        post_grad_v: &GpuBuffer,
        external_grad_v: &GpuBuffer,
        push: &StatePush,
        schedule: StateBackwardSchedule,
    ) -> Result<()> {
        let groups = u32::try_from(
            batch
                .checked_mul(self.heads)
                .context("RWKV fused state-backward workgroup count overflow")?,
        )
        .context("RWKV fused state-backward workgroup count exceeds Vulkan u32 range")?;
        let geometry = self.selected_backward_kernel_geometry(batch);

        match schedule {
            StateBackwardSchedule::RkvAdd3 => {
                let kernel = self
                    .state_backward_kernel_for_geometry(schedule, geometry)
                    .with_context(|| {
                        format!(
                            "RWKV rkv-add3 state-backward kernel geometry {} is unavailable",
                            geometry.label()
                        )
                    })?;
                kernel.record_dispatch(
                    commands,
                    &[
                        state,
                        &self.new_state,
                        &self.r,
                        &self.scaled_k,
                        &self.v,
                        &self.kk,
                        a,
                        w,
                        &self.saved_sa,
                        grad_new_state,
                        grad_tmix,
                        post_grad_r,
                        post_grad_k,
                        post_grad_v,
                        external_grad_v,
                        &self.grad_state,
                        &self.total_grad_v_external,
                        &self.total_grad_r,
                        &self.total_grad_scaled_k,
                        &self.grad_kk,
                        &self.grad_a_direct,
                        &self.grad_w,
                    ],
                    bytemuck::bytes_of(push),
                    [groups, 1, 1],
                )?;
            }
            StateBackwardSchedule::RkvAdd3KeyTransform => {
                let kernel = self
                    .state_backward_kernel_for_geometry(schedule, geometry)
                    .with_context(|| {
                        format!(
                            "RWKV rkv-add3-key state-backward kernel geometry {} is unavailable",
                            geometry.label()
                        )
                    })?;
                kernel.record_dispatch(
                    commands,
                    &[
                        state,
                        &self.new_state,
                        &self.r,
                        &self.scaled_k,
                        &self.v,
                        &self.kk,
                        a,
                        w,
                        &self.saved_sa,
                        grad_new_state,
                        grad_tmix,
                        post_grad_r,
                        post_grad_k,
                        post_grad_v,
                        external_grad_v,
                        &self.grad_state,
                        &self.total_grad_v_external,
                        &self.total_grad_r,
                        &self.total_grad_scaled_k,
                        &self.grad_kk,
                        &self.grad_a_direct,
                        &self.grad_w,
                        &self.raw_k,
                        &self.k_k,
                        &self.k_a,
                        &self.grad_raw_k,
                        &self.grad_a,
                        &self.grad_k_k_partial,
                        &self.grad_k_a_partial,
                    ],
                    bytemuck::bytes_of(push),
                    [groups, 1, 1],
                )?;
            }
            StateBackwardSchedule::RkvAdd3KeyTransformReduce => {
                if batch != 1 {
                    bail!("RWKV rkv-add3-key-reduce schedule is valid only for batch=1");
                }
                let kernel = self
                    .state_backward_kernel_for_geometry(schedule, geometry)
                    .with_context(|| {
                        format!(
                            "RWKV rkv-add3-key-reduce state-backward kernel geometry {} is unavailable",
                            geometry.label()
                        )
                    })?;
                kernel.record_dispatch(
                    commands,
                    &[
                        state,
                        &self.new_state,
                        &self.r,
                        &self.scaled_k,
                        &self.v,
                        &self.kk,
                        a,
                        w,
                        &self.saved_sa,
                        grad_new_state,
                        grad_tmix,
                        post_grad_r,
                        post_grad_k,
                        post_grad_v,
                        external_grad_v,
                        &self.grad_state,
                        &self.total_grad_v_external,
                        &self.total_grad_r,
                        &self.total_grad_scaled_k,
                        &self.grad_kk,
                        &self.grad_a_direct,
                        &self.grad_w,
                        &self.raw_k,
                        &self.k_k,
                        &self.k_a,
                        &self.grad_raw_k,
                        &self.grad_a,
                        &self.grad_k_k_partial,
                        &self.grad_k_a_partial,
                        &self.grad_k_k,
                        &self.grad_k_a,
                    ],
                    bytemuck::bytes_of(push),
                    [groups, 1, 1],
                )?;
            }
        }

        if schedule == StateBackwardSchedule::RkvAdd3 {
            self.record_key_transform_backward(commands, batch, a, &self.total_grad_scaled_k)?;
        }

        if schedule != StateBackwardSchedule::RkvAdd3KeyTransformReduce {
            let mix_push = MixPush {
                batch: batch as u32,
                width: self.width as u32,
            };
            self.key_transform_param_reduce.record_dispatch(
                commands,
                &[
                    &self.grad_k_k_partial,
                    &self.grad_k_a_partial,
                    &self.grad_k_k,
                    &self.grad_k_a,
                ],
                bytemuck::bytes_of(&mix_push),
                [div_ceil_u32(self.width, 64), 1, 1],
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_state_backward_packed_accumulating_rkv_external(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        packed_state: &GpuBuffer,
        matrix_offset: usize,
        a: &GpuBuffer,
        w: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_tmix: &GpuBuffer,
        post_grad_r: &GpuBuffer,
        post_grad_k: &GpuBuffer,
        post_grad_v: &GpuBuffer,
        external_grad_v: &GpuBuffer,
    ) -> Result<()> {
        let kernel = self
            .state_backward_fused_rkv_add3_key_transform_packed
            .as_ref()
            .context("RWKV packed state-backward kernel is unavailable")?;
        let groups = u32::try_from(
            batch
                .checked_mul(self.heads)
                .context("RWKV packed state-backward workgroup count overflow")?,
        )
        .context("RWKV packed state-backward workgroup count exceeds Vulkan u32 range")?;
        let push = PackedStatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
            matrix_offset: matrix_offset as u32,
        };
        kernel.record_dispatch(
            commands,
            &[
                packed_state,
                &self.new_state,
                &self.r,
                &self.scaled_k,
                &self.v,
                &self.kk,
                a,
                w,
                &self.saved_sa,
                grad_new_state,
                grad_tmix,
                post_grad_r,
                post_grad_k,
                post_grad_v,
                external_grad_v,
                &self.grad_state,
                &self.total_grad_v_external,
                &self.total_grad_r,
                &self.total_grad_scaled_k,
                &self.grad_kk,
                &self.grad_a_direct,
                &self.grad_w,
                &self.raw_k,
                &self.k_k,
                &self.k_a,
                &self.grad_raw_k,
                &self.grad_a,
                &self.grad_k_k_partial,
                &self.grad_k_a_partial,
            ],
            bytemuck::bytes_of(&push),
            [groups, 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_state_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_tmix: &GpuBuffer,
        push: &StatePush,
        vector_groups: [u32; 3],
    ) -> Result<()> {
        if self.head_size <= 64 {
            let groups = u32::try_from(
                batch
                    .checked_mul(self.heads)
                    .context("RWKV fused state-backward workgroup count overflow")?,
            )
            .context("RWKV fused state-backward workgroup count exceeds Vulkan u32 range")?;
            return self.state_backward_fused.record_dispatch(
                commands,
                &[
                    state,
                    &self.new_state,
                    &self.r,
                    &self.scaled_k,
                    &self.v,
                    &self.kk,
                    a,
                    w,
                    &self.saved_sa,
                    grad_new_state,
                    grad_tmix,
                    &self.grad_state,
                    &self.grad_v,
                    &self.grad_r,
                    &self.grad_scaled_k,
                    &self.grad_kk,
                    &self.grad_a_direct,
                    &self.grad_w,
                ],
                bytemuck::bytes_of(push),
                [groups, 1, 1],
            );
        }

        self.state_backward_rows.record_dispatch(
            commands,
            &[
                state,
                &self.r,
                &self.scaled_k,
                &self.kk,
                a,
                w,
                grad_new_state,
                grad_tmix,
                &self.grad_state,
                &self.grad_v,
                &self.saved_q,
            ],
            bytemuck::bytes_of(push),
            vector_groups,
        )?;
        self.state_backward_cols.record_dispatch(
            commands,
            &[
                state,
                &self.new_state,
                &self.r,
                &self.scaled_k,
                &self.v,
                &self.kk,
                a,
                w,
                &self.saved_sa,
                &self.saved_q,
                grad_new_state,
                grad_tmix,
                &self.grad_r,
                &self.grad_scaled_k,
                &self.grad_kk,
                &self.grad_a_direct,
                &self.grad_w,
            ],
            bytemuck::bytes_of(push),
            vector_groups,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_state_backward_accumulating_rk(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_tmix: &GpuBuffer,
        post_grad_r: &GpuBuffer,
        post_grad_k: &GpuBuffer,
        push: &StatePush,
        vector_groups: [u32; 3],
    ) -> Result<bool> {
        if self.head_size <= 64 {
            if let Some(kernel) = &self.state_backward_fused_rk_add {
                let groups = u32::try_from(
                    batch
                        .checked_mul(self.heads)
                        .context("RWKV fused state-backward workgroup count overflow")?,
                )
                .context("RWKV fused state-backward workgroup count exceeds Vulkan u32 range")?;
                kernel.record_dispatch(
                    commands,
                    &[
                        state,
                        &self.new_state,
                        &self.r,
                        &self.scaled_k,
                        &self.v,
                        &self.kk,
                        a,
                        w,
                        &self.saved_sa,
                        grad_new_state,
                        grad_tmix,
                        post_grad_r,
                        post_grad_k,
                        &self.grad_state,
                        &self.grad_v,
                        &self.total_grad_r,
                        &self.total_grad_scaled_k,
                        &self.grad_kk,
                        &self.grad_a_direct,
                        &self.grad_w,
                    ],
                    bytemuck::bytes_of(push),
                    [groups, 1, 1],
                )?;
                return Ok(true);
            }
        }

        self.record_state_backward(
            commands,
            batch,
            state,
            a,
            w,
            grad_new_state,
            grad_tmix,
            push,
            vector_groups,
        )?;
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_state_backward_accumulating_rkv_external(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_tmix: &GpuBuffer,
        post_grad_r: &GpuBuffer,
        post_grad_k: &GpuBuffer,
        post_grad_v: &GpuBuffer,
        external_grad_v: &GpuBuffer,
        push: &StatePush,
    ) -> Result<(bool, bool, bool)> {
        if self.head_size > 64 {
            return Ok((false, false, false));
        }
        let decision = if batch == 1 {
            self.state_backward_schedule_batch1
        } else {
            self.state_backward_schedule_multi
        };
        let Some(decision) = decision else {
            return Ok((false, false, false));
        };

        self.record_state_backward_schedule(
            commands,
            batch,
            state,
            a,
            w,
            grad_new_state,
            grad_tmix,
            post_grad_r,
            post_grad_k,
            post_grad_v,
            external_grad_v,
            push,
            decision.schedule,
        )?;
        // Every measured candidate records the complete recurrence + key
        // transform + key-parameter reduction chain. The tuple describes which
        // downstream work has already been recorded, not how many dispatches
        // the chosen schedule used internally.
        Ok((true, true, true))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward(
        &mut self,
        batch: usize,
        state: &[f32],
        x_norm: &[f32],
        previous: &[f32],
        a: &[f32],
        w: &[f32],
        grad_new_state: &[f32],
        grad_tmix: &[f32],
    ) -> Result<RwkvTimeMixCoreResult> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV time-mix core batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV vector size overflow")?;
        let state_len = vector_len
            .checked_mul(self.head_size)
            .context("RWKV state size overflow")?;
        validate_len("state", state, state_len)?;
        validate_len("x_norm", x_norm, vector_len)?;
        validate_len("previous", previous, vector_len)?;
        validate_len("a", a, vector_len)?;
        validate_len("w", w, vector_len)?;
        validate_len("grad_new_state", grad_new_state, state_len)?;
        validate_len("grad_tmix", grad_tmix, vector_len)?;

        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let key_push = KeyPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let vector_groups = [div_ceil_u32(vector_len, 64), 1, 1];
        let channel_groups = [div_ceil_u32(self.width, 64), 1, 1];
        let head_groups = [div_ceil_u32(batch * self.heads, 64), 1, 1];
        let linear_forward_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let linear_weight_groups = [
            div_ceil_u32(self.width, 16),
            div_ceil_u32(self.width, 16),
            1,
        ];

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.state, state)?;
        commands.upload_f32(&self.x_norm, x_norm)?;
        commands.upload_f32(&self.previous, previous)?;
        commands.upload_f32(&self.a, a)?;
        commands.upload_f32(&self.w, w)?;
        commands.upload_f32(&self.grad_new_state, grad_new_state)?;
        commands.upload_f32(&self.grad_tmix, grad_tmix)?;

        self.record_time_mix_projection_key_state_forward(
            &mut commands,
            batch,
            &self.state,
            &self.x_norm,
            &self.previous,
            &self.a,
            &self.w,
            &mix_push,
            &linear_push,
            &key_push,
            &state_push,
            linear_forward_groups,
            head_groups,
            vector_groups,
        )?;
        self.record_state_backward(
            &mut commands,
            batch,
            &self.state,
            &self.a,
            &self.w,
            &self.grad_new_state,
            &self.grad_tmix,
            &state_push,
            vector_groups,
        )?;
        self.record_key_transform_backward(&mut commands, batch, &self.a, &self.grad_scaled_k)?;
        self.key_transform_param_reduce.record_dispatch(
            &mut commands,
            &[
                &self.grad_k_k_partial,
                &self.grad_k_a_partial,
                &self.grad_k_k,
                &self.grad_k_a,
            ],
            bytemuck::bytes_of(&mix_push),
            channel_groups,
        )?;

        self.record_projection_weight_grad(
            &mut commands,
            &self.grad_r,
            &self.grad_raw_k,
            &self.grad_v,
            &linear_push,
            linear_weight_groups,
        )?;
        self.record_projection_time_mix_backward(
            &mut commands,
            &self.x_norm,
            &self.previous,
            &self.grad_r,
            &self.grad_raw_k,
            &self.grad_v,
            &mix_push,
            &linear_push,
            linear_forward_groups,
            channel_groups,
        )?;

        let weight_len = self.width * self.width;
        commands.readback_f32(&self.new_state, &self.new_state_readback, state_len)?;
        commands.readback_f32(&self.tmix, &self.tmix_readback, vector_len)?;
        commands.readback_f32(&self.scaled_k, &self.scaled_k_readback, vector_len)?;
        commands.readback_f32(&self.kk, &self.kk_readback, vector_len)?;
        commands.readback_f32(&self.grad_state, &self.grad_state_readback, state_len)?;
        commands.readback_f32(&self.grad_x_norm, &self.grad_x_norm_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_previous,
            &self.grad_previous_readback,
            vector_len,
        )?;
        commands.readback_f32(&self.grad_a, &self.grad_a_readback, vector_len)?;
        commands.readback_f32(&self.grad_w, &self.grad_w_readback, vector_len)?;
        commands.readback_f32(&self.grad_mix_r, &self.grad_mix_r_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_k, &self.grad_mix_k_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_v, &self.grad_mix_v_readback, self.width)?;
        commands.readback_f32(
            &self.grad_receptance_weight,
            &self.grad_receptance_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(
            &self.grad_key_weight,
            &self.grad_key_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(
            &self.grad_value_weight,
            &self.grad_value_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(&self.grad_k_k, &self.grad_k_k_readback, self.width)?;
        commands.readback_f32(&self.grad_k_a, &self.grad_k_a_readback, self.width)?;
        commands.submit()?;

        Ok(RwkvTimeMixCoreResult {
            new_state: self.new_state_readback.read_f32(state_len)?,
            tmix: self.tmix_readback.read_f32(vector_len)?,
            scaled_k: self.scaled_k_readback.read_f32(vector_len)?,
            kk: self.kk_readback.read_f32(vector_len)?,
            grad_state: self.grad_state_readback.read_f32(state_len)?,
            grad_x_norm: self.grad_x_norm_readback.read_f32(vector_len)?,
            grad_previous: self.grad_previous_readback.read_f32(vector_len)?,
            grad_a: self.grad_a_readback.read_f32(vector_len)?,
            grad_w: self.grad_w_readback.read_f32(vector_len)?,
            grad_mix_r: self.grad_mix_r_readback.read_f32(self.width)?,
            grad_mix_k: self.grad_mix_k_readback.read_f32(self.width)?,
            grad_mix_v: self.grad_mix_v_readback.read_f32(self.width)?,
            grad_receptance_weight: self.grad_receptance_weight_readback.read_f32(weight_len)?,
            grad_key_weight: self.grad_key_weight_readback.read_f32(weight_len)?,
            grad_value_weight: self.grad_value_weight_readback.read_f32(weight_len)?,
            grad_k_k: self.grad_k_k_readback.read_f32(self.width)?,
            grad_k_a: self.grad_k_a_readback.read_f32(self.width)?,
        })
    }

    /// Execute the joined a/w/g + r/k/v + matrix-state graph in one Vulkan
    /// command buffer and queue submission. Unlike `forward_backward`, this
    /// path has no externally supplied a/w tensors: both are produced by the
    /// attached low-rank graph and consumed directly by the key/state kernels.
    ///
    /// `grad_g` remains an explicit terminal gradient until the post-recurrence
    /// GroupNorm/bonus/output-projection slice is attached; keeping the g buffer
    /// resident here is the hand-off point for that next fused stage.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward_fused(
        &mut self,
        batch: usize,
        state: &[f32],
        x_norm: &[f32],
        previous: &[f32],
        grad_new_state: &[f32],
        grad_tmix: &[f32],
        grad_g: &[f32],
    ) -> Result<RwkvFusedTimeMixCoreResult> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV fused time-mix batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let low_rank = self.low_rank.as_ref().context(
            "RWKV fused time-mix requires low-rank parameters; construct with from_model_package_fused",
        )?;
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV fused vector size overflow")?;
        let state_len = vector_len
            .checked_mul(self.head_size)
            .context("RWKV fused state size overflow")?;
        validate_len("state", state, state_len)?;
        validate_len("x_norm", x_norm, vector_len)?;
        validate_len("previous", previous, vector_len)?;
        validate_len("grad_new_state", grad_new_state, state_len)?;
        validate_len("grad_tmix", grad_tmix, vector_len)?;
        validate_len("grad_g", grad_g, vector_len)?;

        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let key_push = KeyPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let vector_groups = [div_ceil_u32(vector_len, 64), 1, 1];
        let channel_groups = [div_ceil_u32(self.width, 64), 1, 1];
        let head_groups = [div_ceil_u32(batch * self.heads, 64), 1, 1];
        let linear_forward_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let linear_weight_groups = [
            div_ceil_u32(self.width, 16),
            div_ceil_u32(self.width, 16),
            1,
        ];

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.state, state)?;
        commands.upload_f32(&self.x_norm, x_norm)?;
        commands.upload_f32(&self.previous, previous)?;
        commands.upload_f32(&self.grad_new_state, grad_new_state)?;
        commands.upload_f32(&self.grad_tmix, grad_tmix)?;
        commands.upload_f32(&self.grad_g_external, grad_g)?;

        low_rank.record_forward(&mut commands, batch, &self.x_norm, &self.previous)?;

        self.record_time_mix_projection_key_state_forward(
            &mut commands,
            batch,
            &self.state,
            &self.x_norm,
            &self.previous,
            low_rank.a_buffer(),
            low_rank.w_buffer(),
            &mix_push,
            &linear_push,
            &key_push,
            &state_push,
            linear_forward_groups,
            head_groups,
            vector_groups,
        )?;
        self.record_state_backward(
            &mut commands,
            batch,
            &self.state,
            low_rank.a_buffer(),
            low_rank.w_buffer(),
            &self.grad_new_state,
            &self.grad_tmix,
            &state_push,
            vector_groups,
        )?;
        self.record_key_transform_backward(
            &mut commands,
            batch,
            low_rank.a_buffer(),
            &self.grad_scaled_k,
        )?;
        self.key_transform_param_reduce.record_dispatch(
            &mut commands,
            &[
                &self.grad_k_k_partial,
                &self.grad_k_a_partial,
                &self.grad_k_k,
                &self.grad_k_a,
            ],
            bytemuck::bytes_of(&mix_push),
            channel_groups,
        )?;

        self.record_projection_weight_grad(
            &mut commands,
            &self.grad_r,
            &self.grad_raw_k,
            &self.grad_v,
            &linear_push,
            linear_weight_groups,
        )?;
        self.record_projection_time_mix_backward(
            &mut commands,
            &self.x_norm,
            &self.previous,
            &self.grad_r,
            &self.grad_raw_k,
            &self.grad_v,
            &mix_push,
            &linear_push,
            linear_forward_groups,
            channel_groups,
        )?;

        low_rank.record_backward(
            &mut commands,
            batch,
            &self.x_norm,
            &self.previous,
            &self.grad_a,
            &self.grad_w,
            &self.grad_g_external,
        )?;

        let len_push = LenPush {
            len: vector_len as u32,
        };
        let add_groups = [div_ceil_u32(vector_len, 256), 1, 1];
        self.vector_add.record_dispatch(
            &mut commands,
            &[
                &self.grad_x_norm,
                low_rank.grad_x_norm_buffer(),
                &self.fused_grad_x_norm,
            ],
            bytemuck::bytes_of(&len_push),
            add_groups,
        )?;
        self.vector_add.record_dispatch(
            &mut commands,
            &[
                &self.grad_previous,
                low_rank.grad_previous_buffer(),
                &self.fused_grad_previous,
            ],
            bytemuck::bytes_of(&len_push),
            add_groups,
        )?;

        let weight_len = self.width * self.width;
        commands.readback_f32(&self.new_state, &self.new_state_readback, state_len)?;
        commands.readback_f32(&self.tmix, &self.tmix_readback, vector_len)?;
        commands.readback_f32(&self.scaled_k, &self.scaled_k_readback, vector_len)?;
        commands.readback_f32(&self.kk, &self.kk_readback, vector_len)?;
        commands.readback_f32(&self.grad_state, &self.grad_state_readback, state_len)?;
        commands.readback_f32(&self.grad_x_norm, &self.grad_x_norm_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_previous,
            &self.grad_previous_readback,
            vector_len,
        )?;
        commands.readback_f32(&self.grad_a, &self.grad_a_readback, vector_len)?;
        commands.readback_f32(&self.grad_w, &self.grad_w_readback, vector_len)?;
        commands.readback_f32(&self.grad_mix_r, &self.grad_mix_r_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_k, &self.grad_mix_k_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_v, &self.grad_mix_v_readback, self.width)?;
        commands.readback_f32(
            &self.grad_receptance_weight,
            &self.grad_receptance_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(
            &self.grad_key_weight,
            &self.grad_key_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(
            &self.grad_value_weight,
            &self.grad_value_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(&self.grad_k_k, &self.grad_k_k_readback, self.width)?;
        commands.readback_f32(&self.grad_k_a, &self.grad_k_a_readback, self.width)?;
        commands.readback_f32(
            &self.fused_grad_x_norm,
            &self.fused_grad_x_norm_readback,
            vector_len,
        )?;
        commands.readback_f32(
            &self.fused_grad_previous,
            &self.fused_grad_previous_readback,
            vector_len,
        )?;
        low_rank.record_readback(&mut commands, batch)?;
        commands.submit()?;

        let core = RwkvTimeMixCoreResult {
            new_state: self.new_state_readback.read_f32(state_len)?,
            tmix: self.tmix_readback.read_f32(vector_len)?,
            scaled_k: self.scaled_k_readback.read_f32(vector_len)?,
            kk: self.kk_readback.read_f32(vector_len)?,
            grad_state: self.grad_state_readback.read_f32(state_len)?,
            grad_x_norm: self.grad_x_norm_readback.read_f32(vector_len)?,
            grad_previous: self.grad_previous_readback.read_f32(vector_len)?,
            grad_a: self.grad_a_readback.read_f32(vector_len)?,
            grad_w: self.grad_w_readback.read_f32(vector_len)?,
            grad_mix_r: self.grad_mix_r_readback.read_f32(self.width)?,
            grad_mix_k: self.grad_mix_k_readback.read_f32(self.width)?,
            grad_mix_v: self.grad_mix_v_readback.read_f32(self.width)?,
            grad_receptance_weight: self.grad_receptance_weight_readback.read_f32(weight_len)?,
            grad_key_weight: self.grad_key_weight_readback.read_f32(weight_len)?,
            grad_value_weight: self.grad_value_weight_readback.read_f32(weight_len)?,
            grad_k_k: self.grad_k_k_readback.read_f32(self.width)?,
            grad_k_a: self.grad_k_a_readback.read_f32(self.width)?,
        };
        let low_rank = low_rank.read_result(batch)?;
        Ok(RwkvFusedTimeMixCoreResult {
            core,
            low_rank,
            grad_x_norm: self.fused_grad_x_norm_readback.read_f32(vector_len)?,
            grad_previous: self.fused_grad_previous_readback.read_f32(vector_len)?,
        })
    }

    /// Execute the complete time-mix training graph through the RWKV output
    /// projection in one Vulkan command buffer. The only terminal gradients are
    /// the recurrent state edge and the projected time-mix output edge.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward_full(
        &mut self,
        batch: usize,
        state: &[f32],
        x_norm: &[f32],
        previous: &[f32],
        grad_new_state: &[f32],
        grad_output: &[f32],
    ) -> Result<RwkvFullTimeMixResult> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV full time-mix batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let low_rank = self.low_rank.as_ref().context(
            "RWKV full time-mix requires low-rank parameters; construct with from_model_package_full",
        )?;
        let post_mix = self.post_mix.as_ref().context(
            "RWKV full time-mix requires post-mix parameters; construct with from_model_package_full",
        )?;
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV full vector size overflow")?;
        let state_len = vector_len
            .checked_mul(self.head_size)
            .context("RWKV full state size overflow")?;
        validate_len("state", state, state_len)?;
        validate_len("x_norm", x_norm, vector_len)?;
        validate_len("previous", previous, vector_len)?;
        validate_len("grad_new_state", grad_new_state, state_len)?;
        validate_len("grad_output", grad_output, vector_len)?;

        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let key_push = KeyPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let vector_groups = [div_ceil_u32(vector_len, 64), 1, 1];
        let channel_groups = [div_ceil_u32(self.width, 64), 1, 1];
        let head_groups = [div_ceil_u32(batch * self.heads, 64), 1, 1];
        let linear_forward_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let linear_weight_groups = [
            div_ceil_u32(self.width, 16),
            div_ceil_u32(self.width, 16),
            1,
        ];
        let len_push = LenPush {
            len: vector_len as u32,
        };
        let add_groups = [div_ceil_u32(vector_len, 256), 1, 1];

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.state, state)?;
        commands.upload_f32(&self.x_norm, x_norm)?;
        commands.upload_f32(&self.previous, previous)?;
        commands.upload_f32(&self.grad_new_state, grad_new_state)?;
        post_mix.upload_grad_output(&mut commands, grad_output)?;

        low_rank.record_forward(&mut commands, batch, &self.x_norm, &self.previous)?;
        self.record_time_mix_projection_key_state_forward(
            &mut commands,
            batch,
            &self.state,
            &self.x_norm,
            &self.previous,
            low_rank.a_buffer(),
            low_rank.w_buffer(),
            &mix_push,
            &linear_push,
            &key_push,
            &state_push,
            linear_forward_groups,
            head_groups,
            vector_groups,
        )?;

        post_mix.record_forward(
            &mut commands,
            batch,
            &self.tmix,
            &self.r,
            &self.scaled_k,
            &self.v,
            low_rank.g_buffer(),
        )?;
        post_mix.record_backward(
            &mut commands,
            batch,
            &self.tmix,
            &self.r,
            &self.scaled_k,
            &self.v,
            low_rank.g_buffer(),
            post_mix.grad_output_buffer(),
        )?;

        self.record_state_backward(
            &mut commands,
            batch,
            &self.state,
            low_rank.a_buffer(),
            low_rank.w_buffer(),
            &self.grad_new_state,
            post_mix.grad_tmix_buffer(),
            &state_push,
            vector_groups,
        )?;

        self.vector_add.record_dispatch(
            &mut commands,
            &[&self.grad_r, post_mix.grad_r_buffer(), &self.total_grad_r],
            bytemuck::bytes_of(&len_push),
            add_groups,
        )?;
        self.vector_add.record_dispatch(
            &mut commands,
            &[
                &self.grad_scaled_k,
                post_mix.grad_k_buffer(),
                &self.total_grad_scaled_k,
            ],
            bytemuck::bytes_of(&len_push),
            add_groups,
        )?;
        self.vector_add.record_dispatch(
            &mut commands,
            &[&self.grad_v, post_mix.grad_v_buffer(), &self.total_grad_v],
            bytemuck::bytes_of(&len_push),
            add_groups,
        )?;

        self.record_key_transform_backward(
            &mut commands,
            batch,
            low_rank.a_buffer(),
            &self.total_grad_scaled_k,
        )?;
        self.key_transform_param_reduce.record_dispatch(
            &mut commands,
            &[
                &self.grad_k_k_partial,
                &self.grad_k_a_partial,
                &self.grad_k_k,
                &self.grad_k_a,
            ],
            bytemuck::bytes_of(&mix_push),
            channel_groups,
        )?;

        self.record_projection_weight_grad(
            &mut commands,
            &self.total_grad_r,
            &self.grad_raw_k,
            &self.total_grad_v,
            &linear_push,
            linear_weight_groups,
        )?;
        self.record_projection_time_mix_backward(
            &mut commands,
            &self.x_norm,
            &self.previous,
            &self.total_grad_r,
            &self.grad_raw_k,
            &self.total_grad_v,
            &mix_push,
            &linear_push,
            linear_forward_groups,
            channel_groups,
        )?;

        low_rank.record_backward(
            &mut commands,
            batch,
            &self.x_norm,
            &self.previous,
            &self.grad_a,
            &self.grad_w,
            post_mix.grad_g_buffer(),
        )?;
        self.vector_add.record_dispatch(
            &mut commands,
            &[
                &self.grad_x_norm,
                low_rank.grad_x_norm_buffer(),
                &self.fused_grad_x_norm,
            ],
            bytemuck::bytes_of(&len_push),
            add_groups,
        )?;
        self.vector_add.record_dispatch(
            &mut commands,
            &[
                &self.grad_previous,
                low_rank.grad_previous_buffer(),
                &self.fused_grad_previous,
            ],
            bytemuck::bytes_of(&len_push),
            add_groups,
        )?;

        let weight_len = self.width * self.width;
        commands.readback_f32(&self.new_state, &self.new_state_readback, state_len)?;
        commands.readback_f32(&self.tmix, &self.tmix_readback, vector_len)?;
        commands.readback_f32(&self.scaled_k, &self.scaled_k_readback, vector_len)?;
        commands.readback_f32(&self.kk, &self.kk_readback, vector_len)?;
        commands.readback_f32(&self.grad_state, &self.grad_state_readback, state_len)?;
        commands.readback_f32(&self.grad_x_norm, &self.grad_x_norm_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_previous,
            &self.grad_previous_readback,
            vector_len,
        )?;
        commands.readback_f32(&self.grad_a, &self.grad_a_readback, vector_len)?;
        commands.readback_f32(&self.grad_w, &self.grad_w_readback, vector_len)?;
        commands.readback_f32(&self.grad_mix_r, &self.grad_mix_r_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_k, &self.grad_mix_k_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_v, &self.grad_mix_v_readback, self.width)?;
        commands.readback_f32(
            &self.grad_receptance_weight,
            &self.grad_receptance_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(
            &self.grad_key_weight,
            &self.grad_key_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(
            &self.grad_value_weight,
            &self.grad_value_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(&self.grad_k_k, &self.grad_k_k_readback, self.width)?;
        commands.readback_f32(&self.grad_k_a, &self.grad_k_a_readback, self.width)?;
        commands.readback_f32(
            &self.fused_grad_x_norm,
            &self.fused_grad_x_norm_readback,
            vector_len,
        )?;
        commands.readback_f32(
            &self.fused_grad_previous,
            &self.fused_grad_previous_readback,
            vector_len,
        )?;
        low_rank.record_readback(&mut commands, batch)?;
        post_mix.record_readback(&mut commands, batch)?;
        commands.submit()?;

        let core = RwkvTimeMixCoreResult {
            new_state: self.new_state_readback.read_f32(state_len)?,
            tmix: self.tmix_readback.read_f32(vector_len)?,
            scaled_k: self.scaled_k_readback.read_f32(vector_len)?,
            kk: self.kk_readback.read_f32(vector_len)?,
            grad_state: self.grad_state_readback.read_f32(state_len)?,
            grad_x_norm: self.grad_x_norm_readback.read_f32(vector_len)?,
            grad_previous: self.grad_previous_readback.read_f32(vector_len)?,
            grad_a: self.grad_a_readback.read_f32(vector_len)?,
            grad_w: self.grad_w_readback.read_f32(vector_len)?,
            grad_mix_r: self.grad_mix_r_readback.read_f32(self.width)?,
            grad_mix_k: self.grad_mix_k_readback.read_f32(self.width)?,
            grad_mix_v: self.grad_mix_v_readback.read_f32(self.width)?,
            grad_receptance_weight: self.grad_receptance_weight_readback.read_f32(weight_len)?,
            grad_key_weight: self.grad_key_weight_readback.read_f32(weight_len)?,
            grad_value_weight: self.grad_value_weight_readback.read_f32(weight_len)?,
            grad_k_k: self.grad_k_k_readback.read_f32(self.width)?,
            grad_k_a: self.grad_k_a_readback.read_f32(self.width)?,
        };
        let low_rank = low_rank.read_result(batch)?;
        let post_mix = post_mix.read_result(batch)?;
        Ok(RwkvFullTimeMixResult {
            core,
            low_rank,
            post_mix,
            grad_x_norm: self.fused_grad_x_norm_readback.read_f32(vector_len)?,
            grad_previous: self.fused_grad_previous_readback.read_f32(vector_len)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_full_forward_with_residual(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        residual: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<()> {
        self.record_full_forward_inner(
            commands,
            batch,
            state,
            x_norm,
            previous,
            Some((residual, output)),
            None,
        )
    }

    pub(crate) fn can_fuse_layer_norm_forward(&self) -> bool {
        self.low_rank
            .as_ref()
            .is_some_and(RwkvLowRankOp::can_fuse_layer_norm_forward)
    }

    pub(crate) fn can_record_packed_forward_only(&self) -> bool {
        (self
            .time_mix_linear3_key_state_forward_packed_fast
            .is_some()
            || [2u32, 4, 8].into_iter().any(|rows| {
                self.time_mix_linear3_key_state_forward_packed_fast_weight_reuse
                    .rows(rows)
                    .is_some_and(|kernels| !kernels.is_empty())
            }))
            && self.can_fuse_layer_norm_forward()
            && self.post_mix.is_some()
    }

    pub(crate) fn can_record_packed_backward_rematerialization(&self) -> bool {
        self.time_mix_linear3_key_state_forward_packed_tape
            .is_some()
            && self
                .state_backward_fused_rkv_add3_key_transform_packed
                .is_some()
            && self.can_fuse_layer_norm_forward()
            && self.post_mix.is_some()
    }

    /// Rebuild the complete backward tape directly from a packed recurrent
    /// history slot. This deliberately avoids materializing the dense matrix
    /// state while preserving the legacy forward arithmetic consumed by the
    /// existing reverse graph.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_packed_backward_rematerialization_with_layer_norm_residual(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        packed_state: &GpuBuffer,
        matrix_offset: usize,
        x: &GpuBuffer,
        ln_weight: &GpuBuffer,
        ln_bias: &GpuBuffer,
        x_norm: &GpuBuffer,
        ln_mean: &GpuBuffer,
        ln_rstd: &GpuBuffer,
        previous_tm: &GpuBuffer,
        residual: &GpuBuffer,
        output: &GpuBuffer,
        eps: f32,
    ) -> Result<bool> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV packed backward rematerialization batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        if matrix_offset < 3 {
            bail!("RWKV packed backward rematerialization matrix offset must be at least 3");
        }
        let Some(kernel) = self.time_mix_linear3_key_state_forward_packed_tape.as_ref() else {
            return Ok(false);
        };
        let low_rank = self
            .low_rank
            .as_ref()
            .context("RWKV packed backward rematerialization requires low-rank parameters")?;
        let post_mix = self
            .post_mix
            .as_ref()
            .context("RWKV packed backward rematerialization requires post-mix parameters")?;
        if !low_rank.can_fuse_layer_norm_forward() {
            return Ok(false);
        }

        low_rank.record_forward_from_layer_norm(
            commands,
            batch,
            x,
            ln_weight,
            ln_bias,
            x_norm,
            ln_mean,
            ln_rstd,
            previous_tm,
            eps,
        )?;
        let push = PackedStatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
            matrix_offset: matrix_offset as u32,
        };
        let groups = u32::try_from(
            batch
                .checked_mul(self.heads)
                .context("RWKV packed backward rematerialization workgroup count overflow")?,
        )
        .context(
            "RWKV packed backward rematerialization workgroup count exceeds Vulkan u32 range",
        )?;
        kernel.record_dispatch(
            commands,
            &[
                x_norm,
                packed_state,
                &self.mix_r,
                &self.mix_k,
                &self.mix_v,
                &self.receptance_weight,
                &self.key_weight,
                &self.value_weight,
                low_rank.a_buffer(),
                &self.k_k,
                &self.k_a,
                low_rank.w_buffer(),
                &self.xr,
                &self.xk,
                &self.xv,
                &self.r,
                &self.raw_k,
                &self.v,
                &self.scaled_k,
                &self.kk,
                &self.new_state,
                &self.tmix,
                &self.saved_sa,
                previous_tm,
            ],
            bytemuck::bytes_of(&push),
            [groups, 1, 1],
        )?;
        post_mix.record_forward_optional_residual(
            commands,
            batch,
            &self.tmix,
            &self.r,
            &self.scaled_k,
            &self.v,
            low_rank.g_buffer(),
            Some((residual, output)),
        )?;
        Ok(true)
    }

    /// Record the first TBPTT forward transition directly against the packed
    /// recurrent history. The reverse pass rematerializes the complete cell
    /// forward, so this path intentionally retains only the values required to
    /// produce the residual output and next packed state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_packed_forward_only_with_layer_norm_residual(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        packed_state: &GpuBuffer,
        matrix_offset: usize,
        state_clamp: f32,
        x: &GpuBuffer,
        ln_weight: &GpuBuffer,
        ln_bias: &GpuBuffer,
        x_norm: &GpuBuffer,
        ln_mean: &GpuBuffer,
        ln_rstd: &GpuBuffer,
        previous_tm: &GpuBuffer,
        residual: &GpuBuffer,
        output: &GpuBuffer,
        packed_new_state: &GpuBuffer,
        eps: f32,
    ) -> Result<bool> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV packed forward-only batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        if matrix_offset < 3 {
            bail!("RWKV packed forward-only matrix offset must be at least 3");
        }
        if !state_clamp.is_finite() || state_clamp < 0.0 {
            bail!("RWKV packed forward-only state clamp must be finite and non-negative");
        }
        let low_rank = self
            .low_rank
            .as_ref()
            .context("RWKV packed forward-only transition requires low-rank parameters")?;
        let post_mix = self
            .post_mix
            .as_ref()
            .context("RWKV packed forward-only transition requires post-mix parameters")?;
        if !low_rank.can_fuse_layer_norm_forward() {
            return Ok(false);
        }

        let push = PackedFastStatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
            matrix_offset: matrix_offset as u32,
            state_clamp,
        };
        let decision = self.choose_forward_projection_topology(
            batch,
            ForwardProjectionRecurrence::Packed,
            |topology| {
                self.time_packed_forward_projection_topology_ms(
                    batch,
                    packed_state,
                    previous_tm,
                    low_rank.a_buffer(),
                    low_rank.w_buffer(),
                    packed_new_state,
                    &push,
                    topology,
                )
            },
        )?;

        low_rank.record_forward_from_layer_norm(
            commands,
            batch,
            x,
            ln_weight,
            ln_bias,
            x_norm,
            ln_mean,
            ln_rstd,
            previous_tm,
            eps,
        )?;
        self.record_packed_forward_projection_topology(
            commands,
            batch,
            x_norm,
            packed_state,
            previous_tm,
            low_rank.a_buffer(),
            low_rank.w_buffer(),
            packed_new_state,
            &push,
            decision.topology,
        )?;

        post_mix.record_forward_optional_residual(
            commands,
            batch,
            &self.tmix,
            &self.r,
            &self.scaled_k,
            &self.v,
            low_rank.g_buffer(),
            Some((residual, output)),
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_packed_forward_projection_topology(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        packed_state: &GpuBuffer,
        previous_tm: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        packed_new_state: &GpuBuffer,
        push: &PackedFastStatePush,
        topology: ForwardProjectionTopology,
    ) -> Result<()> {
        let (kernel, projection_groups) = match topology {
            ForwardProjectionTopology::Baseline => {
                let kernel = self
                    .time_mix_linear3_key_state_forward_packed_fast
                    .as_ref()
                    .context("RWKV packed baseline forward topology is unavailable")?;
                let groups = batch
                    .checked_mul(self.heads)
                    .context("RWKV packed forward workgroup count overflow")?;
                (kernel, groups)
            }
            ForwardProjectionTopology::WeightReuse { rows, tile } => {
                if batch < rows as usize {
                    bail!("RWKV packed weight-reuse{rows} topology requires batch >= {rows}");
                }
                let kernel = self
                    .time_mix_linear3_key_state_forward_packed_fast_weight_reuse
                    .get(rows, tile)
                    .with_context(|| {
                        format!(
                            "RWKV packed weight-reuse{rows} tile {tile} is unavailable on this Vulkan device"
                        )
                    })?;
                let groups = self
                    .heads
                    .checked_mul(batch.div_ceil(rows as usize))
                    .context("RWKV packed weight-reuse workgroup count overflow")?;
                (kernel, groups)
            }
        };
        let projection_groups = u32::try_from(projection_groups)
            .context("RWKV packed forward workgroup count exceeds Vulkan u32 range")?;
        kernel.record_dispatch(
            commands,
            &[
                x_norm,
                packed_state,
                &self.mix_r,
                &self.mix_k,
                &self.mix_v,
                &self.receptance_weight,
                &self.key_weight,
                &self.value_weight,
                a,
                &self.k_k,
                &self.k_a,
                w,
                &self.r,
                &self.v,
                &self.scaled_k,
                &self.tmix,
                packed_new_state,
                previous_tm,
            ],
            bytemuck::bytes_of(push),
            [projection_groups, 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn time_packed_forward_projection_topology_ms(
        &self,
        batch: usize,
        packed_state: &GpuBuffer,
        previous_tm: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        packed_new_state: &GpuBuffer,
        push: &PackedFastStatePush,
        topology: ForwardProjectionTopology,
    ) -> Result<f64> {
        let repetitions = if batch.saturating_mul(self.heads) >= 128 {
            4
        } else {
            16
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..repetitions {
            self.record_packed_forward_projection_topology(
                &mut commands,
                batch,
                &self.x_norm,
                packed_state,
                previous_tm,
                a,
                w,
                packed_new_state,
                push,
                topology,
            )?;
        }
        let started = Instant::now();
        commands.submit()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_full_forward_with_layer_norm_residual(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        x: &GpuBuffer,
        ln_weight: &GpuBuffer,
        ln_bias: &GpuBuffer,
        x_norm: &GpuBuffer,
        ln_mean: &GpuBuffer,
        ln_rstd: &GpuBuffer,
        previous: &GpuBuffer,
        residual: &GpuBuffer,
        output: &GpuBuffer,
        eps: f32,
    ) -> Result<()> {
        self.record_full_forward_inner(
            commands,
            batch,
            state,
            x_norm,
            previous,
            Some((residual, output)),
            Some(LayerNormForwardInput {
                x,
                weight: ln_weight,
                bias: ln_bias,
                mean: ln_mean,
                rstd: ln_rstd,
                eps,
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_full_forward_inner(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        residual_output: Option<(&GpuBuffer, &GpuBuffer)>,
        layer_norm: Option<LayerNormForwardInput<'_>>,
    ) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV full time-mix batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let low_rank = self.low_rank.as_ref().context(
            "RWKV full time-mix requires low-rank parameters; construct with from_model_package_full",
        )?;
        let post_mix = self.post_mix.as_ref().context(
            "RWKV full time-mix requires post-mix parameters; construct with from_model_package_full",
        )?;
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV full vector size overflow")?;
        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let key_push = KeyPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let vector_groups = [div_ceil_u32(vector_len, 64), 1, 1];
        let head_groups = [div_ceil_u32(batch * self.heads, 64), 1, 1];
        let linear_forward_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];

        if let Some(norm) = layer_norm {
            low_rank.record_forward_from_layer_norm(
                commands,
                batch,
                norm.x,
                norm.weight,
                norm.bias,
                x_norm,
                norm.mean,
                norm.rstd,
                previous,
                norm.eps,
            )?;
        } else {
            low_rank.record_forward(commands, batch, x_norm, previous)?;
        }
        self.record_time_mix_projection_key_state_forward(
            commands,
            batch,
            state,
            x_norm,
            previous,
            low_rank.a_buffer(),
            low_rank.w_buffer(),
            &mix_push,
            &linear_push,
            &key_push,
            &state_push,
            linear_forward_groups,
            head_groups,
            vector_groups,
        )?;

        post_mix.record_forward_optional_residual(
            commands,
            batch,
            &self.tmix,
            &self.r,
            &self.scaled_k,
            &self.v,
            low_rank.g_buffer(),
            residual_output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_full_backward_with_v_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_v_external: Option<&GpuBuffer>,
        grad_x_norm_external: Option<&GpuBuffer>,
    ) -> Result<bool> {
        self.record_full_backward_with_v_grad_inner(
            commands,
            batch,
            state,
            None,
            x_norm,
            previous,
            grad_new_state,
            grad_output,
            grad_v_external,
            grad_x_norm_external,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_full_backward_with_v_grad_from_packed_state(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        packed_state: &GpuBuffer,
        matrix_offset: usize,
        dense_state_scratch: &GpuBuffer,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_v_external: Option<&GpuBuffer>,
        grad_x_norm_external: Option<&GpuBuffer>,
    ) -> Result<bool> {
        if !self.can_record_packed_backward_rematerialization() {
            bail!("RWKV packed backward rematerialization is unavailable on this Vulkan device");
        }
        self.record_full_backward_with_v_grad_inner(
            commands,
            batch,
            dense_state_scratch,
            Some((packed_state, matrix_offset)),
            x_norm,
            previous,
            grad_new_state,
            grad_output,
            grad_v_external,
            grad_x_norm_external,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_full_backward_with_v_grad_inner(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        state: &GpuBuffer,
        packed_state: Option<(&GpuBuffer, usize)>,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_new_state: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_v_external: Option<&GpuBuffer>,
        grad_x_norm_external: Option<&GpuBuffer>,
    ) -> Result<bool> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV full time-mix batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let low_rank = self.low_rank.as_ref().context(
            "RWKV full time-mix requires low-rank parameters; construct with from_model_package_full",
        )?;
        let post_mix = self.post_mix.as_ref().context(
            "RWKV full time-mix requires post-mix parameters; construct with from_model_package_full",
        )?;
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV full vector size overflow")?;
        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        let state_push = StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let vector_groups = [div_ceil_u32(vector_len, 64), 1, 1];
        let channel_groups = [div_ceil_u32(self.width, 64), 1, 1];
        let linear_forward_groups = [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1];
        let linear_weight_groups = [
            div_ceil_u32(self.width, 16),
            div_ceil_u32(self.width, 16),
            1,
        ];
        let len_push = LenPush {
            len: vector_len as u32,
        };
        let add_groups = [div_ceil_u32(vector_len, 256), 1, 1];

        post_mix.record_backward(
            commands,
            batch,
            &self.tmix,
            &self.r,
            &self.scaled_k,
            &self.v,
            low_rank.g_buffer(),
            grad_output,
        )?;

        let segment_decision = if batch == 1 {
            self.backward_segment_schedule_batch1
        } else {
            self.backward_segment_schedule_multi
        };
        let mut backward_segment_recorded = false;
        let (state_accumulated_rkv, key_transform_recorded, key_param_reduce_recorded) =
            if let Some(external_grad_v) = grad_v_external {
                if let Some((packed_state, matrix_offset)) = packed_state {
                    self.record_state_backward_packed_accumulating_rkv_external(
                        commands,
                        batch,
                        packed_state,
                        matrix_offset,
                        low_rank.a_buffer(),
                        low_rank.w_buffer(),
                        grad_new_state,
                        post_mix.grad_tmix_buffer(),
                        post_mix.grad_r_buffer(),
                        post_mix.grad_k_buffer(),
                        post_mix.grad_v_buffer(),
                        external_grad_v,
                    )?;
                    (true, true, false)
                } else if let Some(decision) = segment_decision {
                    self.record_backward_segment_schedule(
                        commands,
                        batch,
                        state,
                        low_rank.a_buffer(),
                        low_rank.w_buffer(),
                        grad_new_state,
                        post_mix.grad_tmix_buffer(),
                        post_mix.grad_r_buffer(),
                        post_mix.grad_k_buffer(),
                        post_mix.grad_v_buffer(),
                        external_grad_v,
                        x_norm,
                        previous,
                        &state_push,
                        &mix_push,
                        &linear_push,
                        linear_forward_groups,
                        linear_weight_groups,
                        channel_groups,
                        decision.schedule,
                    )?;
                    backward_segment_recorded = true;
                    (true, true, true)
                } else {
                    self.record_state_backward_accumulating_rkv_external(
                        commands,
                        batch,
                        state,
                        low_rank.a_buffer(),
                        low_rank.w_buffer(),
                        grad_new_state,
                        post_mix.grad_tmix_buffer(),
                        post_mix.grad_r_buffer(),
                        post_mix.grad_k_buffer(),
                        post_mix.grad_v_buffer(),
                        external_grad_v,
                        &state_push,
                    )?
                }
            } else {
                (false, false, false)
            };
        let state_accumulated_rk = if state_accumulated_rkv {
            true
        } else {
            self.record_state_backward_accumulating_rk(
                commands,
                batch,
                state,
                low_rank.a_buffer(),
                low_rank.w_buffer(),
                grad_new_state,
                post_mix.grad_tmix_buffer(),
                post_mix.grad_r_buffer(),
                post_mix.grad_k_buffer(),
                &state_push,
                vector_groups,
            )?
        };

        if !state_accumulated_rk {
            self.vector_add.record_dispatch(
                commands,
                &[&self.grad_r, post_mix.grad_r_buffer(), &self.total_grad_r],
                bytemuck::bytes_of(&len_push),
                add_groups,
            )?;
            self.vector_add.record_dispatch(
                commands,
                &[
                    &self.grad_scaled_k,
                    post_mix.grad_k_buffer(),
                    &self.total_grad_scaled_k,
                ],
                bytemuck::bytes_of(&len_push),
                add_groups,
            )?;
        }
        let total_grad_v = if state_accumulated_rkv {
            &self.total_grad_v_external
        } else if let Some(external) = grad_v_external {
            self.vector_add3.record_dispatch(
                commands,
                &[
                    &self.grad_v,
                    post_mix.grad_v_buffer(),
                    external,
                    &self.total_grad_v_external,
                ],
                bytemuck::bytes_of(&len_push),
                add_groups,
            )?;
            &self.total_grad_v_external
        } else {
            self.vector_add.record_dispatch(
                commands,
                &[&self.grad_v, post_mix.grad_v_buffer(), &self.total_grad_v],
                bytemuck::bytes_of(&len_push),
                add_groups,
            )?;
            &self.total_grad_v
        };

        if !key_transform_recorded {
            self.record_key_transform_backward(
                commands,
                batch,
                low_rank.a_buffer(),
                &self.total_grad_scaled_k,
            )?;
        }
        if !key_param_reduce_recorded {
            self.key_transform_param_reduce.record_dispatch(
                commands,
                &[
                    &self.grad_k_k_partial,
                    &self.grad_k_a_partial,
                    &self.grad_k_k,
                    &self.grad_k_a,
                ],
                bytemuck::bytes_of(&mix_push),
                channel_groups,
            )?;
        }

        if !backward_segment_recorded {
            self.record_projection_weight_grad(
                commands,
                &self.total_grad_r,
                &self.grad_raw_k,
                total_grad_v,
                &linear_push,
                linear_weight_groups,
            )?;
            self.record_projection_time_mix_backward(
                commands,
                x_norm,
                previous,
                &self.total_grad_r,
                &self.grad_raw_k,
                total_grad_v,
                &mix_push,
                &linear_push,
                linear_forward_groups,
                channel_groups,
            )?;
        }

        let selected_low_rank_fan_in =
            segment_decision.and_then(|decision| decision.schedule.low_rank_fan_in);
        let (low_rank_accumulated, external_x_accumulated) =
            if let Some(fan_in_schedule) = selected_low_rank_fan_in {
                low_rank.record_backward_with_fan_in_schedule_and_workgroup_size(
                    commands,
                    batch,
                    x_norm,
                    previous,
                    &self.grad_a,
                    &self.grad_w,
                    post_mix.grad_g_buffer(),
                    &self.grad_x_norm,
                    &self.grad_previous,
                    grad_x_norm_external,
                    &self.fused_grad_x_norm,
                    &self.fused_grad_previous,
                    fan_in_schedule,
                    self.selected_backward_kernel_geometry(batch)
                        .workgroup_size() as usize,
                )?
            } else if grad_x_norm_external.is_some() {
                low_rank.record_backward_accumulating_outer_x(
                    commands,
                    batch,
                    x_norm,
                    previous,
                    &self.grad_a,
                    &self.grad_w,
                    post_mix.grad_g_buffer(),
                    &self.grad_x_norm,
                    &self.grad_previous,
                    grad_x_norm_external,
                    &self.fused_grad_x_norm,
                    &self.fused_grad_previous,
                )?
            } else {
                (
                    low_rank.record_backward_accumulating(
                        commands,
                        batch,
                        x_norm,
                        previous,
                        &self.grad_a,
                        &self.grad_w,
                        post_mix.grad_g_buffer(),
                        &self.grad_x_norm,
                        &self.grad_previous,
                        &self.fused_grad_x_norm,
                        &self.fused_grad_previous,
                    )?,
                    false,
                )
            };
        if !low_rank_accumulated {
            self.vector_add.record_dispatch(
                commands,
                &[
                    &self.grad_x_norm,
                    low_rank.grad_x_norm_buffer(),
                    &self.fused_grad_x_norm,
                ],
                bytemuck::bytes_of(&len_push),
                add_groups,
            )?;
            self.vector_add.record_dispatch(
                commands,
                &[
                    &self.grad_previous,
                    low_rank.grad_previous_buffer(),
                    &self.fused_grad_previous,
                ],
                bytemuck::bytes_of(&len_push),
                add_groups,
            )?;
        }
        Ok(external_x_accumulated)
    }

    pub(crate) fn new_state_buffer(&self) -> &GpuBuffer {
        &self.new_state
    }

    pub(crate) fn grad_state_buffer(&self) -> &GpuBuffer {
        &self.grad_state
    }

    pub(crate) fn full_grad_x_norm_buffer(&self) -> &GpuBuffer {
        &self.fused_grad_x_norm
    }

    pub(crate) fn full_grad_previous_buffer(&self) -> &GpuBuffer {
        &self.fused_grad_previous
    }

    pub(crate) fn value_buffer(&self) -> &GpuBuffer {
        &self.v
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn head_size(&self) -> usize {
        self.head_size
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub(crate) fn trainables(&self) -> Result<Vec<RwkvTrainableRef<'_>>> {
        let decay = RwkvDecayClass::Decay;
        let mut trainables = vec![
            RwkvTrainableRef {
                name: "x_r",
                parameter: &self.mix_r,
                gradient: &self.grad_mix_r,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "x_k",
                parameter: &self.mix_k,
                gradient: &self.grad_mix_k,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "x_v",
                parameter: &self.mix_v,
                gradient: &self.grad_mix_v,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "receptance.weight",
                parameter: &self.receptance_weight,
                gradient: &self.grad_receptance_weight,
                len: self.width * self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "key.weight",
                parameter: &self.key_weight,
                gradient: &self.grad_key_weight,
                len: self.width * self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "value.weight",
                parameter: &self.value_weight,
                gradient: &self.grad_value_weight,
                len: self.width * self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "k_k",
                parameter: &self.k_k,
                gradient: &self.grad_k_k,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "k_a",
                parameter: &self.k_a,
                gradient: &self.grad_k_a,
                len: self.width,
                decay_class: decay,
            },
        ];
        trainables.extend(
            self.low_rank
                .as_ref()
                .context("full RWKV trainable registry requires low-rank parameters")?
                .trainables(),
        );
        trainables.extend(
            self.post_mix
                .as_ref()
                .context("full RWKV trainable registry requires post-mix parameters")?
                .trainables(),
        );
        Ok(trainables)
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn numerics_policy(&self) -> RwkvNumericsPolicy {
        self.numerics_policy
    }

    pub(crate) fn supports_numerics_policy(&self, policy: RwkvNumericsPolicy) -> bool {
        match policy {
            RwkvNumericsPolicy::StrictParity => true,
            RwkvNumericsPolicy::FastSubgroup => self.key_transform_backward_subgroup.is_some(),
            RwkvNumericsPolicy::FastRecurrentTree => {
                BackwardKernelGeometry::ALL.into_iter().any(|geometry| {
                    if !self.recurrent_parallel_geometry_changes_order(geometry) {
                        return false;
                    }
                    match geometry {
                        BackwardKernelGeometry::Wg32 => self
                            .state_backward_fused_rkv_add3_key_transform_tree_wg32
                            .is_some(),
                        BackwardKernelGeometry::Wg64 => self
                            .state_backward_fused_rkv_add3_key_transform_tree
                            .is_some(),
                        BackwardKernelGeometry::Wg128 => self
                            .state_backward_fused_rkv_add3_key_transform_tree_wg128
                            .is_some(),
                    }
                })
            }
            RwkvNumericsPolicy::FastRecurrentTiled => {
                BackwardKernelGeometry::ALL.into_iter().any(|geometry| {
                    if !self.recurrent_parallel_geometry_changes_order(geometry) {
                        return false;
                    }
                    match geometry {
                        BackwardKernelGeometry::Wg32 => self
                            .state_backward_fused_rkv_add3_key_transform_tiled_wg32
                            .is_some(),
                        BackwardKernelGeometry::Wg64 => self
                            .state_backward_fused_rkv_add3_key_transform_tiled
                            .is_some(),
                        BackwardKernelGeometry::Wg128 => self
                            .state_backward_fused_rkv_add3_key_transform_tiled_wg128
                            .is_some(),
                    }
                })
            }
            RwkvNumericsPolicy::FastRecurrentSubgroup => {
                BackwardKernelGeometry::ALL.into_iter().any(|geometry| {
                    if !self.recurrent_subgroup_geometry_changes_order(geometry) {
                        return false;
                    }
                    match geometry {
                        BackwardKernelGeometry::Wg32 => self
                            .state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg32
                            .is_some(),
                        BackwardKernelGeometry::Wg64 => self
                            .state_backward_fused_rkv_add3_key_transform_recurrent_subgroup
                            .is_some(),
                        BackwardKernelGeometry::Wg128 => self
                            .state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg128
                            .is_some(),
                    }
                })
            }
        }
    }

    /// Select the RWKV backward numerical contract. Strict parity is the
    /// default. Experimental modes are opt-in and fail closed when the selected
    /// Vulkan device/model geometry cannot execute the requested reduction.
    pub fn set_numerics_policy(&mut self, policy: RwkvNumericsPolicy) -> Result<()> {
        if !self.supports_numerics_policy(policy) {
            match policy {
                RwkvNumericsPolicy::StrictParity => unreachable!("strict parity is always supported"),
                RwkvNumericsPolicy::FastSubgroup => {
                    let caps = self.device.subgroup_capabilities();
                    bail!(
                        "RWKV fast-subgroup numerics are unavailable on device={} subgroup={} compute={} basic={} arithmetic={}",
                        self.device.name(),
                        caps.subgroup_size,
                        caps.compute_supported,
                        caps.basic_supported,
                        caps.arithmetic_supported
                    );
                }
                RwkvNumericsPolicy::FastRecurrentTree => bail!(
                    "RWKV fast-recurrent-tree numerics are unavailable on device={} head_size={}; no compiled workgroup geometry can split each recurrent row/column across multiple lanes",
                    self.device.name(),
                    self.head_size
                ),
                RwkvNumericsPolicy::FastRecurrentTiled => bail!(
                    "RWKV fast-recurrent-tiled numerics are unavailable on device={} head_size={}; no compiled workgroup geometry can assign contiguous recurrent tiles to multiple lanes",
                    self.device.name(),
                    self.head_size
                ),
                RwkvNumericsPolicy::FastRecurrentSubgroup => {
                    let caps = self.device.subgroup_capabilities();
                    bail!(
                        "RWKV fast-recurrent-subgroup numerics are unavailable on device={} subgroup={} head_size={} compute={} basic={} arithmetic={}; the policy requires subgroup arithmetic and a compiled complete-subgroup workgroup able to cover the head with one or two hardware subgroups",
                        self.device.name(),
                        caps.subgroup_size,
                        self.head_size,
                        caps.compute_supported,
                        caps.basic_supported,
                        caps.arithmetic_supported
                    )
                }
            }
        }
        self.numerics_policy = policy;
        Ok(())
    }

    /// Runtime-selected external-V matrix-state backward schedule. The choice
    /// changes Vulkan dispatch topology only; model/checkpoint tensor layouts
    /// remain identical to the PyTorch/CUDA path.
    pub fn state_backward_schedule_label(&self, batch: usize) -> Option<&'static str> {
        let decision = if batch == 1 {
            self.state_backward_schedule_batch1
        } else if batch > 1 && batch <= self.max_batch {
            self.state_backward_schedule_multi
        } else {
            None
        }?;
        Some(decision.schedule.label())
    }

    pub fn state_backward_schedule_was_autotuned(&self, batch: usize) -> bool {
        let decision = if batch == 1 {
            self.state_backward_schedule_batch1
        } else if batch > 1 && batch <= self.max_batch {
            self.state_backward_schedule_multi
        } else {
            None
        };
        decision.is_some_and(|choice| choice.autotuned)
    }

    pub fn state_backward_profile_batch(&self) -> usize {
        self.state_backward_profile_batch
    }

    /// Runtime-selected recurrence + projection + low-rank fan-in backward
    /// segment used by the
    /// external-V full training graph. This is the hardware-measured schedule
    /// that spans matrix-state recurrence, key transform/reduction, projection
    /// weight gradients, projection input gradients, a/w/g backward, and the
    /// shared normalized-input/previous-token fan-in.
    pub fn backward_segment_schedule_label(&self, batch: usize) -> Option<String> {
        let decision = if batch == 1 {
            self.backward_segment_schedule_batch1
        } else if batch > 1 && batch <= self.max_batch {
            self.backward_segment_schedule_multi
        } else {
            None
        }?;
        Some(decision.schedule.label())
    }

    pub fn backward_segment_schedule_was_autotuned(&self, batch: usize) -> bool {
        let decision = if batch == 1 {
            self.backward_segment_schedule_batch1
        } else if batch > 1 && batch <= self.max_batch {
            self.backward_segment_schedule_multi
        } else {
            None
        };
        decision.is_some_and(|choice| choice.autotuned)
    }

    /// Stable label for the compiled local-size variant used by the head-owned
    /// fused state-backward and low-rank shared-input fan-in kernels. Geometry
    /// changes dispatch occupancy only; tensor layouts and serial reduction
    /// order remain unchanged.
    pub fn backward_kernel_geometry_label(&self, batch: usize) -> Option<&'static str> {
        if batch == 0 || batch > self.max_batch {
            return None;
        }
        let selected = self.selected_backward_kernel_geometry(batch);
        self.available_backward_kernel_geometries(batch)
            .contains(&selected)
            .then_some(selected.label())
    }

    pub fn available_backward_kernel_geometry_labels(&self, batch: usize) -> Result<Vec<String>> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-kernel geometry batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        Ok(self
            .available_backward_kernel_geometries(batch)
            .into_iter()
            .map(|geometry| geometry.label().to_owned())
            .collect())
    }

    pub(crate) fn available_backward_kernel_geometry_labels_for_numerics(
        &self,
        batch: usize,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<Vec<String>> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-kernel geometry batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        Ok(self
            .available_backward_kernel_geometries_for_numerics(batch, numerics_policy)
            .into_iter()
            .map(|geometry| geometry.label().to_owned())
            .collect())
    }

    /// Check whether one independently selected full-cell schedule and
    /// backward-kernel geometry can execute together under the requested
    /// numerics contract. This is the safety gate used by the outer factorized
    /// scheduler before it composes a schedule/geometry pair that may never
    /// have appeared as an exact historical profile arm.
    pub(crate) fn backward_segment_schedule_geometry_pair_available(
        &self,
        batch: usize,
        schedule_label: &str,
        geometry_label: &str,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<bool> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward schedule/geometry batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        let Some(geometry) = self
            .available_backward_kernel_geometries_for_numerics(batch, numerics_policy)
            .into_iter()
            .find(|geometry| geometry.label() == geometry_label)
        else {
            return Ok(false);
        };
        let Some(schedule) = self
            .available_backward_segment_schedules(batch, true)?
            .into_iter()
            .find(|schedule| schedule.label() == schedule_label)
        else {
            return Ok(false);
        };
        if self
            .state_backward_kernel_for_geometry_with_policy(
                schedule.state,
                geometry,
                numerics_policy,
            )
            .is_none()
        {
            return Ok(false);
        }
        let Some(fan_in) = schedule.low_rank_fan_in else {
            return Ok(true);
        };
        let Some(low_rank) = self.low_rank.as_ref() else {
            return Ok(false);
        };
        Ok(low_rank.backward_fan_in_geometry_available(fan_in, geometry.workgroup_size() as usize))
    }

    /// Force one compiled backward geometry arm for reproducible profiling or
    /// parity experiments. The label must come from
    /// `available_backward_kernel_geometry_labels` for the same batch.
    pub fn set_backward_kernel_geometry_label(&mut self, batch: usize, label: &str) -> Result<()> {
        self.set_backward_kernel_geometry_label_for_numerics(batch, label, self.numerics_policy)
    }

    pub(crate) fn set_backward_kernel_geometry_label_for_numerics(
        &mut self,
        batch: usize,
        label: &str,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-kernel geometry batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        let geometry = self
            .available_backward_kernel_geometries_for_numerics(batch, numerics_policy)
            .into_iter()
            .find(|geometry| geometry.label() == label)
            .with_context(|| {
                format!(
                    "RWKV backward-kernel geometry {label:?} is unavailable for device={} subgroup={} width={} head_size={} batch={batch} numerics={}",
                    self.device.name(),
                    self.device.subgroup_capabilities().subgroup_size,
                    self.width,
                    self.head_size,
                    numerics_policy.label()
                )
            })?;
        if batch == 1 {
            self.backward_kernel_geometry_batch1 = geometry;
        } else {
            self.backward_kernel_geometry_multi = geometry;
        }
        Ok(())
    }

    /// Enumerate executable full-cell backward topologies for this device and
    /// batch geometry. The labels are stable policy identifiers: selecting one
    /// changes only Vulkan dispatch/fusion topology, never tensor/checkpoint
    /// layout.
    pub(crate) fn available_backward_segment_schedule_labels(
        &self,
        batch: usize,
    ) -> Result<Vec<String>> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-segment schedule batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        self.available_backward_segment_schedules(batch, true)
            .map(|schedules| {
                schedules
                    .into_iter()
                    .map(BackwardSegmentSchedule::label)
                    .collect()
            })
    }

    /// Decompose one executable full-cell schedule into the three dispatch
    /// decisions that make up its stable label. Keeping this knowledge beside
    /// `BackwardSegmentSchedule` prevents the outer tape scheduler from
    /// reverse-engineering private enum semantics when it performs marginal
    /// learning over state fusion, projection ordering, and low-rank fan-in.
    pub(crate) fn backward_segment_schedule_factor_labels(
        &self,
        batch: usize,
        label: &str,
    ) -> Result<(String, String, Option<String>)> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-segment schedule batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        let schedule = self
            .available_backward_segment_schedules(batch, true)?
            .into_iter()
            .find(|candidate| candidate.label() == label)
            .with_context(|| {
                format!(
                    "RWKV backward-segment schedule {label:?} is unavailable for device={} width={} head_size={} batch={batch}",
                    self.device.name(), self.width, self.head_size
                )
            })?;
        Ok((
            schedule.state.label().to_owned(),
            schedule.projection.label().to_owned(),
            schedule
                .low_rank_fan_in
                .map(|fan_in| fan_in.label().to_owned()),
        ))
    }

    /// Compose independently selected schedule factors back into an executable
    /// full-cell schedule label. `None` means the Cartesian coordinate is not
    /// compiled/allowed on this device; callers can then keep a known-good
    /// baseline sibling rather than attempting an invalid dispatch plan.
    pub(crate) fn compose_backward_segment_schedule_label(
        &self,
        batch: usize,
        state_label: &str,
        projection_label: &str,
        low_rank_fan_in_label: Option<&str>,
    ) -> Result<Option<String>> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-segment schedule batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        Ok(self
            .available_backward_segment_schedules(batch, true)?
            .into_iter()
            .find(|schedule| {
                schedule.state.label() == state_label
                    && schedule.projection.label() == projection_label
                    && schedule
                        .low_rank_fan_in
                        .map(RwkvLowRankFanInSchedule::label)
                        == low_rank_fan_in_label
            })
            .map(BackwardSegmentSchedule::label))
    }

    /// Enumerate the fusion-depth neighbors of `current_label` that are
    /// executable with one compiled workgroup geometry. Projection ordering
    /// and low-rank fan-in are held fixed, so the outer training-submission
    /// policy can measure the interaction between recurrence fusion depth and
    /// local-size geometry without exploding the full topology Cartesian
    /// product. Tensor layouts and FP32 reduction semantics are unchanged.
    pub(crate) fn backward_segment_fusion_depth_neighbor_labels_for_geometry(
        &self,
        batch: usize,
        current_label: &str,
        geometry_label: &str,
    ) -> Result<Vec<String>> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-segment fusion/geometry batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        let geometry = self
            .available_backward_kernel_geometries(batch)
            .into_iter()
            .find(|geometry| geometry.label() == geometry_label)
            .with_context(|| {
                format!(
                    "RWKV backward-kernel geometry {geometry_label:?} is unavailable for device={} subgroup={} width={} head_size={} batch={batch}",
                    self.device.name(),
                    self.device.subgroup_capabilities().subgroup_size,
                    self.width,
                    self.head_size
                )
            })?;
        let schedules = self.available_backward_segment_schedules(batch, true)?;
        let current = schedules
            .iter()
            .copied()
            .find(|schedule| schedule.label() == current_label)
            .with_context(|| {
                format!(
                    "RWKV backward-segment schedule {current_label:?} is unavailable for device={} width={} head_size={} batch={batch}",
                    self.device.name(), self.width, self.head_size
                )
            })?;
        let low_rank = self
            .low_rank
            .as_ref()
            .context("RWKV full-cell fusion/geometry search requires low-rank a/w/g")?;
        let workgroup_size = geometry.workgroup_size() as usize;

        Ok(schedules
            .into_iter()
            .filter(|schedule| {
                schedule.state != current.state
                    && schedule.projection == current.projection
                    && schedule.low_rank_fan_in == current.low_rank_fan_in
                    && self
                        .state_backward_kernel_for_geometry(schedule.state, geometry)
                        .is_some()
                    && schedule.low_rank_fan_in.is_none_or(|fan_in| {
                        low_rank.backward_fan_in_geometry_available(fan_in, workgroup_size)
                    })
            })
            .map(|schedule| schedule.label())
            .collect())
    }

    /// Install one previously enumerated topology as the live scheduler
    /// choice. This is the mutation seam used by the outer device-aware policy
    /// when it explores a composite arm spanning tape geometry and RWKV fusion
    /// topology.
    pub(crate) fn set_backward_segment_schedule_label(
        &mut self,
        batch: usize,
        label: &str,
    ) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV backward-segment schedule batch {batch} exceeds configured max_batch {}",
                self.max_batch
            );
        }
        let schedule = self
            .available_backward_segment_schedules(batch, true)?
            .into_iter()
            .find(|candidate| candidate.label() == label)
            .with_context(|| {
                format!(
                    "RWKV backward-segment schedule {label:?} is unavailable for device={} width={} head_size={} batch={batch}",
                    self.device.name(), self.width, self.head_size
                )
            })?;
        let decision = BackwardSegmentDecision {
            schedule,
            // The outer contextual policy made this selection rather than the
            // one-time microbenchmark autotuner.
            autotuned: false,
        };
        if batch == 1 {
            self.backward_segment_schedule_batch1 = Some(decision);
        } else {
            self.backward_segment_schedule_multi = Some(decision);
        }
        Ok(())
    }

    /// Low-rank a/w/g geometry loaded from the canonical model package.
    /// Exposing the tuple lets production-training profiling distinguish
    /// otherwise identical width/head geometries without changing any tensor
    /// layout or scheduler decision.
    pub fn low_rank_ranks(&self) -> Option<(usize, usize, usize)> {
        self.low_rank.as_ref().map(RwkvLowRankOp::ranks)
    }

    pub(crate) fn low_rank_fp16_parameter_storage_active(&self) -> bool {
        self.low_rank
            .as_ref()
            .is_some_and(RwkvLowRankOp::fp16_parameter_storage_active)
    }

    pub(crate) fn low_rank_native_fp16_backward_compute_active(&self) -> bool {
        self.low_rank
            .as_ref()
            .is_some_and(RwkvLowRankOp::native_fp16_backward_compute_active)
    }

    pub(crate) fn low_rank_native_fp16_parameter_grad_compute_active(&self) -> bool {
        self.low_rank
            .as_ref()
            .is_some_and(RwkvLowRankOp::native_fp16_parameter_grad_compute_active)
    }

    pub(crate) fn low_rank_parameter_grad_arithmetic(&self) -> RwkvLowRankParameterGradArithmetic {
        self.low_rank
            .as_ref()
            .map_or(RwkvLowRankParameterGradArithmetic::Fp32, |low_rank| {
                low_rank.parameter_grad_arithmetic()
            })
    }

    pub(crate) fn configure_backward_source_scale(
        &mut self,
        source_scale: f32,
        source_scaled_backward_domain: bool,
    ) -> Result<()> {
        if !source_scale.is_finite() || source_scale <= 0.0 {
            bail!("RWKV time-mix backward source scale must be finite and positive");
        }
        self.backward_source_scale = source_scale;
        self.source_scaled_backward_domain = source_scaled_backward_domain;
        if let Some(low_rank) = self.low_rank.as_mut() {
            low_rank
                .configure_backward_source_scale(source_scale, source_scaled_backward_domain)?;
        }
        Ok(())
    }

    /// Whether recurrent r/k/v projection input adjoints execute their multiply
    /// in native Float16 with pre-multiply adjoint lifting and FP32 accumulation.
    /// Canonical parameters and checkpoint gradients remain FP32.
    pub fn projection_native_fp16_backward_compute_active(&self) -> bool {
        self.native_fp16_projection_input_grad
    }

    /// Enable the loss-scaled native-FP16 r/k/v projection dX execution arm.
    /// This deliberately does not require FP16 parameter storage: weights are
    /// cast from the canonical PyTorch-layout FP32 buffers inside the shader so
    /// the checkpoint/interchange boundary is unchanged.
    pub fn enable_projection_native_fp16_backward_compute(&mut self) -> Result<()> {
        if !self
            .device
            .mixed_precision_capabilities()
            .shader_float16_enabled
        {
            bail!("native-FP16 RWKV projection backward requires shaderFloat16");
        }
        self.linear3_input_grad_fp16_scaled
            .as_ref()
            .context("native-FP16 RWKV projection input-gradient kernel was not created")?;
        self.linear3_input_grad_fp16_source_scaled
            .as_ref()
            .context(
                "source-scaled native-FP16 RWKV projection input-gradient kernel was not created",
            )?;
        self.native_fp16_projection_input_grad = true;
        Ok(())
    }

    pub(crate) fn low_rank_fp16_full_forward_first_stage_arm_label(&self) -> Option<&'static str> {
        self.low_rank
            .as_ref()
            .and_then(RwkvLowRankOp::fp16_full_forward_first_stage_arm_label)
    }

    pub(crate) fn install_low_rank_fp16_parameter_mirrors(
        &mut self,
        mirrors: RwkvLowRankFp16ParameterMirrors,
    ) -> Result<()> {
        self.low_rank
            .as_mut()
            .context("RWKV time-mix has no low-rank parameter block to mirror")?
            .install_fp16_parameter_mirrors(mirrors)
    }

    pub(crate) fn enable_low_rank_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.low_rank
            .as_mut()
            .context("RWKV time-mix has no low-rank parameter block for native-FP16 backward")?
            .enable_native_fp16_backward_compute()
    }

    pub(crate) fn enable_low_rank_native_fp16_parameter_grad_compute(
        &mut self,
        widen_product: bool,
        compensated_operands: bool,
    ) -> Result<()> {
        self.low_rank
            .as_mut()
            .context("RWKV time-mix has no low-rank parameter block for native-FP16 dW")?
            .enable_native_fp16_parameter_grad_compute(widen_product, compensated_operands)
    }

    pub fn heads(&self) -> usize {
        self.heads
    }
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "RWKV {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("RWKV {name} contains non-finite values");
    }
    Ok(())
}

fn vector_width(shape: &[usize]) -> Option<usize> {
    match shape {
        [width] if *width > 0 => Some(*width),
        [1, width] if *width > 0 => Some(*width),
        _ => None,
    }
}

fn read_vector_tensor(path: &Path, name: &str, width: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if vector_width(&shape) != Some(width) {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{width}] or [1, {width}]");
    }
    Ok(values)
}

fn read_matrix_tensor(path: &Path, name: &str, width: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if shape != [width, width] {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{width}, {width}]");
    }
    Ok(values)
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

fn forced_state_backward_schedule() -> Result<Option<StateBackwardSchedule>> {
    let raw = match std::env::var("HIERARCHOS_RWKV_STATE_BACKWARD_SCHEDULE") {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(err) => bail!("reading HIERARCHOS_RWKV_STATE_BACKWARD_SCHEDULE: {err}"),
    };
    let normalized = raw.trim().to_ascii_lowercase();
    let schedule = match normalized.as_str() {
        "" | "auto" => None,
        "rkv" | "rkv-add3" => Some(StateBackwardSchedule::RkvAdd3),
        "rkv-key" | "rkv-add3-key" => Some(StateBackwardSchedule::RkvAdd3KeyTransform),
        "rkv-key-reduce" | "rkv-add3-key-reduce" => {
            Some(StateBackwardSchedule::RkvAdd3KeyTransformReduce)
        }
        _ => bail!(
            "HIERARCHOS_RWKV_STATE_BACKWARD_SCHEDULE must be auto, rkv, rkv-key, or rkv-key-reduce; got {raw:?}"
        ),
    };
    Ok(schedule)
}

fn forced_projection_backward_schedule() -> Result<Option<ProjectionBackwardSchedule>> {
    let raw = match std::env::var("HIERARCHOS_RWKV_PROJECTION_BACKWARD_SCHEDULE") {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(err) => bail!("reading HIERARCHOS_RWKV_PROJECTION_BACKWARD_SCHEDULE: {err}"),
    };
    let normalized = raw.trim().to_ascii_lowercase();
    let schedule = match normalized.as_str() {
        "" | "auto" => None,
        "weight-fused" | "weight->fused-input-mix" => {
            Some(ProjectionBackwardSchedule::WeightThenFusedMix)
        }
        "fused-weight" | "fused-input-mix->weight" => {
            Some(ProjectionBackwardSchedule::FusedMixThenWeight)
        }
        "weight-split" | "weight->split-input-mix" => {
            Some(ProjectionBackwardSchedule::WeightThenSplitMix)
        }
        "split-weight" | "split-input-mix->weight" => {
            Some(ProjectionBackwardSchedule::SplitMixThenWeight)
        }
        _ => bail!(
            "HIERARCHOS_RWKV_PROJECTION_BACKWARD_SCHEDULE must be auto, weight-fused, fused-weight, weight-split, or split-weight; got {raw:?}"
        ),
    };
    Ok(schedule)
}

fn forced_low_rank_fan_in_schedule() -> Result<Option<RwkvLowRankFanInSchedule>> {
    let raw = match std::env::var("HIERARCHOS_RWKV_LOW_RANK_FAN_IN_SCHEDULE") {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(err) => bail!("reading HIERARCHOS_RWKV_LOW_RANK_FAN_IN_SCHEDULE: {err}"),
    };
    let normalized = raw.trim().to_ascii_lowercase();
    let schedule = match normalized.as_str() {
        "" | "auto" => None,
        "split" | "split-fan-in" | "low-rank-split-fan-in" => {
            Some(RwkvLowRankFanInSchedule::Split)
        }
        "fused-base" | "base" | "low-rank-fused-base-fan-in" => {
            Some(RwkvLowRankFanInSchedule::FusedBase)
        }
        "fused-outer" | "outer" | "low-rank-fused-outer-fan-in" => {
            Some(RwkvLowRankFanInSchedule::FusedOuter)
        }
        _ => bail!(
            "HIERARCHOS_RWKV_LOW_RANK_FAN_IN_SCHEDULE must be auto, split, fused-base, or fused-outer; got {raw:?}"
        ),
    };
    Ok(schedule)
}

fn backward_segment_persistent_cache_path() -> Option<PathBuf> {
    if std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_DISABLE_PERSISTENT_CACHE").is_some() {
        return None;
    }
    if let Some(path) = std::env::var_os("HIERARCHOS_RWKV_BACKWARD_SEGMENT_CACHE_PATH") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
        return None;
    }
    if let Some(root) = std::env::var_os("LOCALAPPDATA") {
        return Some(
            PathBuf::from(root)
                .join("Hierarchos")
                .join("vulkan-rwkv-backward-segment-v1.json"),
        );
    }
    if let Some(root) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(
            PathBuf::from(root)
                .join("hierarchos")
                .join("vulkan-rwkv-backward-segment-v1.json"),
        );
    }
    std::env::var_os("HOME").map(|root| {
        PathBuf::from(root)
            .join(".cache")
            .join("hierarchos")
            .join("vulkan-rwkv-backward-segment-v1.json")
    })
}

fn load_persistent_backward_segment_schedule(
    key: &BackwardSegmentAutotuneKey,
    candidates: &[BackwardSegmentSchedule],
) -> Result<Option<BackwardSegmentSchedule>> {
    let Some(path) = backward_segment_persistent_cache_path() else {
        return Ok(None);
    };
    let _guard = BACKWARD_SEGMENT_PERSISTENT_CACHE_IO
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| anyhow::anyhow!("RWKV backward-segment persistent cache lock was poisoned"))?;
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("reading {}", path.display())),
    };
    let cache: PersistentBackwardSegmentCache =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if cache.version != BACKWARD_SEGMENT_PERSISTENT_CACHE_VERSION {
        return Ok(None);
    }
    Ok(cache
        .entries
        .iter()
        .rev()
        .find(|entry| &entry.key == key && candidates.contains(&entry.schedule))
        .map(|entry| entry.schedule))
}

fn store_persistent_backward_segment_schedule(
    key: &BackwardSegmentAutotuneKey,
    schedule: BackwardSegmentSchedule,
) -> Result<()> {
    let Some(path) = backward_segment_persistent_cache_path() else {
        return Ok(());
    };
    let _guard = BACKWARD_SEGMENT_PERSISTENT_CACHE_IO
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| anyhow::anyhow!("RWKV backward-segment persistent cache lock was poisoned"))?;
    let mut cache = match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<PersistentBackwardSegmentCache>(&bytes)
            .ok()
            .filter(|cache| cache.version == BACKWARD_SEGMENT_PERSISTENT_CACHE_VERSION)
            .unwrap_or_default(),
        Err(_) => PersistentBackwardSegmentCache::default(),
    };
    if let Some(entry) = cache.entries.iter_mut().find(|entry| entry.key == *key) {
        entry.schedule = schedule;
    } else {
        cache.entries.push(PersistentBackwardSegmentEntry {
            key: key.clone(),
            schedule,
        });
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "creating persistent autotune cache directory {}",
                parent.display()
            )
        })?;
    }
    let bytes =
        serde_json::to_vec_pretty(&cache).context("serializing persistent autotune cache")?;
    fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn fastest_axis_half<T>(scores: HashMap<T, f64>) -> HashSet<T>
where
    T: Copy + Eq + std::hash::Hash,
{
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
    let keep = ranked.len().div_ceil(2).max(1);
    ranked
        .into_iter()
        .take(keep)
        .map(|(value, _)| value)
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ProjectionReuseCoordinate {
    rows: u32,
    tile: u32,
}

/// Build an O(rows + tiles) probe frontier for the projection reuse kernels.
///
/// The compiled shaders form a small `rows x shared-tile` grid. Profiling every
/// pair makes startup cost grow as a Cartesian product and teaches the runtime
/// the same two preferences repeatedly. Instead, anchor on the structurally
/// preferred pair, measure one representative for every row and tile, and let
/// the marginal selector synthesize the promising pair. The synthesized pair
/// is timed once more before it is allowed to beat the portable baseline, so
/// interactions remain measured rather than assumed.
fn factorized_projection_reuse_probe_coordinates(
    candidates: &[ProjectionReuseCoordinate],
    preferred: Option<ProjectionReuseCoordinate>,
) -> Vec<ProjectionReuseCoordinate> {
    let Some(anchor) = preferred
        .filter(|preferred| candidates.contains(preferred))
        .or_else(|| {
            candidates
                .iter()
                .max_by_key(|candidate| (candidate.rows, candidate.tile))
                .copied()
        })
    else {
        return Vec::new();
    };

    let mut rows = candidates
        .iter()
        .map(|candidate| candidate.rows)
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    let mut tiles = candidates
        .iter()
        .map(|candidate| candidate.tile)
        .collect::<Vec<_>>();
    tiles.sort_unstable();
    tiles.dedup();

    let mut probes = Vec::with_capacity(rows.len().saturating_add(tiles.len()));
    let mut push_unique = |coordinate: ProjectionReuseCoordinate| {
        if !probes.contains(&coordinate) {
            probes.push(coordinate);
        }
    };
    push_unique(anchor);

    for row in rows {
        let representative = candidates
            .iter()
            .filter(|candidate| candidate.rows == row)
            .min_by_key(|candidate| {
                (
                    candidate.tile.abs_diff(anchor.tile),
                    u32::MAX - candidate.tile,
                )
            })
            .copied();
        if let Some(representative) = representative {
            push_unique(representative);
        }
    }
    for tile in tiles {
        let representative = candidates
            .iter()
            .filter(|candidate| candidate.tile == tile)
            .min_by_key(|candidate| {
                (
                    candidate.rows.abs_diff(anchor.rows),
                    u32::MAX - candidate.rows,
                )
            })
            .copied();
        if let Some(representative) = representative {
            push_unique(representative);
        }
    }
    probes
}

fn select_factorized_projection_reuse_coordinate(
    candidates: &[ProjectionReuseCoordinate],
    timings: &[(ProjectionReuseCoordinate, f64)],
    anchor: ProjectionReuseCoordinate,
) -> Option<ProjectionReuseCoordinate> {
    if candidates.is_empty() || timings.is_empty() {
        return None;
    }
    let mut row_scores = HashMap::new();
    let mut tile_scores = HashMap::new();
    for &(coordinate, ms) in timings {
        if !ms.is_finite() || ms <= 0.0 {
            continue;
        }
        if coordinate.tile == anchor.tile {
            row_scores
                .entry(coordinate.rows)
                .and_modify(|best: &mut f64| *best = best.min(ms))
                .or_insert(ms);
        }
        if coordinate.rows == anchor.rows {
            tile_scores
                .entry(coordinate.tile)
                .and_modify(|best: &mut f64| *best = best.min(ms))
                .or_insert(ms);
        }
    }

    // A capability-limited grid may not expose the anchor tile for every row
    // (or the anchor row for every tile). Only those missing axes fall back to
    // their measured representative; a dense grid stays a true cross-section
    // comparison and cannot let one factor borrow the other's winning sample.
    for &(coordinate, ms) in timings {
        if !ms.is_finite() || ms <= 0.0 {
            continue;
        }
        row_scores
            .entry(coordinate.rows)
            .and_modify(|best: &mut f64| {
                if !timings.iter().any(|(measured, _)| {
                    measured.rows == coordinate.rows && measured.tile == anchor.tile
                }) {
                    *best = best.min(ms);
                }
            })
            .or_insert(ms);
        tile_scores
            .entry(coordinate.tile)
            .and_modify(|best: &mut f64| {
                if !timings.iter().any(|(measured, _)| {
                    measured.tile == coordinate.tile && measured.rows == anchor.rows
                }) {
                    *best = best.min(ms);
                }
            })
            .or_insert(ms);
    }
    let mut ranked_rows = row_scores
        .iter()
        .map(|(&row, &score)| (row, score))
        .collect::<Vec<_>>();
    ranked_rows.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| rhs.0.cmp(&lhs.0)));
    let mut ranked_tiles = tile_scores
        .iter()
        .map(|(&tile, &score)| (tile, score))
        .collect::<Vec<_>>();
    ranked_tiles.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| rhs.0.cmp(&lhs.0)));
    let (&(best_row, _), &(best_tile, _)) = (ranked_rows.first()?, ranked_tiles.first()?);
    let synthesized = ProjectionReuseCoordinate {
        rows: best_row,
        tile: best_tile,
    };
    if candidates.contains(&synthesized) {
        return Some(synthesized);
    }

    // Sparse capability grids can make the independently best row/tile pair
    // unavailable. In that case choose the available pair with the smallest
    // sum of its measured marginal scores, retaining deterministic larger-axis
    // tie breaks.
    candidates.iter().copied().min_by(|lhs, rhs| {
        let lhs_score = row_scores.get(&lhs.rows).copied().unwrap_or(f64::INFINITY)
            + tile_scores.get(&lhs.tile).copied().unwrap_or(f64::INFINITY);
        let rhs_score = row_scores.get(&rhs.rows).copied().unwrap_or(f64::INFINITY)
            + tile_scores.get(&rhs.tile).copied().unwrap_or(f64::INFINITY);
        lhs_score
            .total_cmp(&rhs_score)
            .then_with(|| rhs.rows.cmp(&lhs.rows))
            .then_with(|| rhs.tile.cmp(&lhs.tile))
    })
}

fn median_interleaved_timings<T, F>(candidates: &[T], mut time_ms: F) -> Result<Vec<(T, f64)>>
where
    T: Copy,
    F: FnMut(T) -> Result<f64>,
{
    let mut samples = vec![Vec::with_capacity(3); candidates.len()];
    for round in 0..3 {
        if round % 2 == 0 {
            for (index, &candidate) in candidates.iter().enumerate() {
                samples[index].push(time_ms(candidate)?);
            }
        } else {
            for (index, &candidate) in candidates.iter().enumerate().rev() {
                samples[index].push(time_ms(candidate)?);
            }
        }
    }
    let mut timings = Vec::with_capacity(candidates.len());
    for (index, &candidate) in candidates.iter().enumerate() {
        samples[index].sort_by(f64::total_cmp);
        timings.push((candidate, samples[index][1]));
    }
    Ok(timings)
}

fn prune_backward_segment_candidates(
    one_shot: &[(BackwardSegmentSchedule, f64)],
    default: BackwardSegmentSchedule,
) -> Vec<BackwardSegmentSchedule> {
    if one_shot.len() <= BACKWARD_SEGMENT_ELIMINATION_THRESHOLD {
        return one_shot.iter().map(|(schedule, _)| *schedule).collect();
    }

    let mut state_scores = HashMap::new();
    let mut projection_scores = HashMap::new();
    let mut low_rank_scores = HashMap::new();
    for &(schedule, ms) in one_shot {
        state_scores
            .entry(schedule.state)
            .and_modify(|best: &mut f64| *best = best.min(ms))
            .or_insert(ms);
        projection_scores
            .entry(schedule.projection)
            .and_modify(|best: &mut f64| *best = best.min(ms))
            .or_insert(ms);
        low_rank_scores
            .entry(schedule.low_rank_fan_in)
            .and_modify(|best: &mut f64| *best = best.min(ms))
            .or_insert(ms);
    }
    let retained_states = fastest_axis_half(state_scores);
    let retained_projections = fastest_axis_half(projection_scores);
    let retained_low_rank = fastest_axis_half(low_rank_scores);
    let one_shot_winner = one_shot
        .iter()
        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
        .map(|(schedule, _)| *schedule)
        .unwrap_or(default);

    one_shot
        .iter()
        .filter_map(|(schedule, _)| {
            let marginal_survivor = retained_states.contains(&schedule.state)
                && retained_projections.contains(&schedule.projection)
                && retained_low_rank.contains(&schedule.low_rank_fan_in);
            (marginal_survivor || *schedule == one_shot_winner || *schedule == default)
                .then_some(*schedule)
        })
        .collect()
}

fn select_state_backward_schedule(
    timings: &[(StateBackwardSchedule, f64)],
    default: StateBackwardSchedule,
) -> StateBackwardSchedule {
    let Some((_, default_ms)) = timings.iter().find(|(schedule, _)| *schedule == default) else {
        return default;
    };
    let mut selected = default;
    let mut selected_ms = *default_ms;
    for &(schedule, ms) in timings {
        // A candidate must win by at least 2% before displacing the deepest
        // compatible fusion. This keeps startup noise from turning equivalent
        // timings into device-dependent schedule churn.
        if ms < selected_ms * 0.98 {
            selected = schedule;
            selected_ms = ms;
        }
    }
    selected
}

fn select_projection_traffic_topology<T>(timings: &[(T, f64)], default: T) -> T
where
    T: Copy + PartialEq,
{
    let Some((_, default_ms)) = timings.iter().find(|(topology, _)| *topology == default) else {
        return default;
    };
    let Some(&(best, best_ms)) = timings.iter().min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1)) else {
        return default;
    };
    // Keep the same 2% displacement margin used by the backward topology
    // tuners. Once an arm clears that margin, choose the actually fastest tile
    // rather than making the answer depend on candidate enumeration order.
    if best_ms < *default_ms * 0.98 {
        best
    } else {
        default
    }
}

fn select_backward_segment_schedule(
    timings: &[(BackwardSegmentSchedule, f64)],
    default: BackwardSegmentSchedule,
) -> BackwardSegmentSchedule {
    let Some((_, default_ms)) = timings.iter().find(|(schedule, _)| *schedule == default) else {
        return default;
    };
    let mut selected = default;
    let mut selected_ms = *default_ms;
    for &(schedule, ms) in timings {
        if ms < selected_ms * 0.98 {
            selected = schedule;
            selected_ms = ms;
        }
    }
    selected
}

#[cfg(test)]
mod state_backward_schedule_tests {
    use crate::vulkan::{ComputeBatch, GpuBuffer};
    use anyhow::Result;

    use super::{
        backward_kernel_geometry_priority, factorized_projection_reuse_probe_coordinates,
        prune_backward_segment_candidates, recurrent_subgroup_geometry_supported,
        select_backward_segment_schedule, select_factorized_projection_reuse_coordinate,
        select_projection_traffic_topology, select_state_backward_schedule, BackwardKernelGeometry,
        BackwardSegmentAutotuneKey, BackwardSegmentSchedule, ForwardProjectionAutotuneKey,
        ForwardProjectionRecurrence, ForwardProjectionTopology, PersistentBackwardSegmentCache,
        PersistentBackwardSegmentEntry, ProjectionBackwardSchedule, ProjectionInputGradTopology,
        ProjectionReuseCoordinate, ProjectionWeightGradTopology, RwkvLowRankFanInSchedule,
        RwkvNumericsPolicy, RwkvTimeMixCoreOp, StateBackwardSchedule, VulkanDevice,
        BACKWARD_SEGMENT_PERSISTENT_CACHE_VERSION,
    };

    #[test]
    fn numerics_policy_labels_include_recurrent_fast_arms() {
        assert_eq!(
            RwkvNumericsPolicy::from_label("fast-recurrent-tiled"),
            Some(RwkvNumericsPolicy::FastRecurrentTiled)
        );
        assert_eq!(
            RwkvNumericsPolicy::FastRecurrentTiled.label(),
            "fast-recurrent-tiled"
        );
        assert_eq!(
            RwkvNumericsPolicy::from_label("tiled"),
            Some(RwkvNumericsPolicy::FastRecurrentTiled)
        );
        assert_eq!(
            RwkvNumericsPolicy::from_label("fast-recurrent-subgroup"),
            Some(RwkvNumericsPolicy::FastRecurrentSubgroup)
        );
        assert_eq!(
            RwkvNumericsPolicy::FastRecurrentSubgroup.label(),
            "fast-recurrent-subgroup"
        );
        assert_eq!(
            RwkvNumericsPolicy::from_label("subgroup-recurrent"),
            Some(RwkvNumericsPolicy::FastRecurrentSubgroup)
        );
    }

    #[test]
    fn backward_kernel_geometry_labels_are_stable() {
        assert_eq!(
            BackwardKernelGeometry::ALL.map(BackwardKernelGeometry::label),
            [
                "rwkv-state-bwd-wg32",
                "rwkv-state-bwd-wg64",
                "rwkv-state-bwd-wg128",
            ]
        );
    }

    #[test]
    fn backward_kernel_geometry_bootstrap_prefers_subgroup_alignment() {
        let mut subgroup_64 = BackwardKernelGeometry::ALL;
        subgroup_64.sort_by_key(|&geometry| backward_kernel_geometry_priority(geometry, 64));
        assert_eq!(
            subgroup_64,
            [
                BackwardKernelGeometry::Wg64,
                BackwardKernelGeometry::Wg128,
                BackwardKernelGeometry::Wg32,
            ]
        );

        let mut subgroup_32 = BackwardKernelGeometry::ALL;
        subgroup_32.sort_by_key(|&geometry| backward_kernel_geometry_priority(geometry, 32));
        assert_eq!(
            subgroup_32,
            [
                BackwardKernelGeometry::Wg32,
                BackwardKernelGeometry::Wg64,
                BackwardKernelGeometry::Wg128,
            ]
        );
    }

    #[test]
    fn recurrent_subgroup_geometry_accepts_complete_multi_wave_workgroups() {
        assert!(recurrent_subgroup_geometry_supported(64, 64, 64));
        assert!(recurrent_subgroup_geometry_supported(128, 64, 64));
        assert!(!recurrent_subgroup_geometry_supported(32, 64, 64));

        assert!(recurrent_subgroup_geometry_supported(64, 32, 32));
        assert!(recurrent_subgroup_geometry_supported(128, 32, 32));
        assert!(recurrent_subgroup_geometry_supported(64, 32, 64));
        assert!(recurrent_subgroup_geometry_supported(128, 32, 64));
        assert!(!recurrent_subgroup_geometry_supported(32, 32, 64));
        assert!(recurrent_subgroup_geometry_supported(64, 32, 33));
        assert!(!recurrent_subgroup_geometry_supported(128, 32, 65));

        assert!(!recurrent_subgroup_geometry_supported(128, 0, 32));
        assert!(!recurrent_subgroup_geometry_supported(96, 64, 32));
        assert!(!recurrent_subgroup_geometry_supported(128, 64, 0));
    }

    #[test]
    fn state_backward_selector_keeps_deep_fusion_on_near_tie() {
        let default = StateBackwardSchedule::RkvAdd3KeyTransformReduce;
        let timings = [
            (StateBackwardSchedule::RkvAdd3, 0.990),
            (StateBackwardSchedule::RkvAdd3KeyTransform, 1.000),
            (default, 1.000),
        ];
        assert_eq!(select_state_backward_schedule(&timings, default), default);
    }

    #[test]
    fn state_backward_selector_accepts_materially_faster_shallow_schedule() {
        let default = StateBackwardSchedule::RkvAdd3KeyTransformReduce;
        let timings = [
            (StateBackwardSchedule::RkvAdd3, 0.900),
            (StateBackwardSchedule::RkvAdd3KeyTransform, 0.970),
            (default, 1.000),
        ];
        assert_eq!(
            select_state_backward_schedule(&timings, default),
            StateBackwardSchedule::RkvAdd3
        );
    }

    #[test]
    fn forward_projection_selector_keeps_baseline_on_near_tie() {
        let baseline = ForwardProjectionTopology::Baseline;
        let timings = [
            (baseline, 1.000),
            (
                ForwardProjectionTopology::WeightReuse { rows: 2, tile: 16 },
                0.991,
            ),
            (
                ForwardProjectionTopology::WeightReuse { rows: 2, tile: 32 },
                0.985,
            ),
        ];
        assert_eq!(
            select_projection_traffic_topology(&timings, baseline),
            baseline
        );
    }

    #[test]
    fn forward_projection_selector_chooses_fastest_material_reuse_tile() {
        let baseline = ForwardProjectionTopology::Baseline;
        let tile64 = ForwardProjectionTopology::WeightReuse { rows: 2, tile: 64 };
        let timings = [
            (baseline, 1.000),
            (
                ForwardProjectionTopology::WeightReuse { rows: 2, tile: 16 },
                0.940,
            ),
            (
                ForwardProjectionTopology::WeightReuse { rows: 2, tile: 32 },
                0.925,
            ),
            (tile64, 0.910),
        ];
        assert_eq!(
            select_projection_traffic_topology(&timings, baseline),
            tile64
        );
    }

    #[test]
    fn projection_reuse_factorization_probes_axes_instead_of_cartesian_grid() {
        let candidates = [2u32, 4, 8]
            .into_iter()
            .flat_map(|rows| {
                [16u32, 32, 64]
                    .into_iter()
                    .map(move |tile| ProjectionReuseCoordinate { rows, tile })
            })
            .collect::<Vec<_>>();
        let preferred = ProjectionReuseCoordinate { rows: 8, tile: 64 };
        let probes = factorized_projection_reuse_probe_coordinates(&candidates, Some(preferred));

        assert_eq!(candidates.len(), 9);
        assert_eq!(probes.len(), 5);
        assert_eq!(probes[0], preferred);
        for rows in [2u32, 4, 8] {
            assert!(probes.iter().any(|probe| probe.rows == rows));
        }
        for tile in [16u32, 32, 64] {
            assert!(probes.iter().any(|probe| probe.tile == tile));
        }
        assert!(!probes.contains(&ProjectionReuseCoordinate { rows: 4, tile: 32 }));
    }

    #[test]
    fn projection_reuse_factorization_composes_unmeasured_rows_tile_pair() {
        let candidates = [2u32, 4, 8]
            .into_iter()
            .flat_map(|rows| {
                [16u32, 32, 64]
                    .into_iter()
                    .map(move |tile| ProjectionReuseCoordinate { rows, tile })
            })
            .collect::<Vec<_>>();
        let timings = [
            (ProjectionReuseCoordinate { rows: 8, tile: 64 }, 1.00),
            (ProjectionReuseCoordinate { rows: 2, tile: 64 }, 0.95),
            (ProjectionReuseCoordinate { rows: 4, tile: 64 }, 0.80),
            (ProjectionReuseCoordinate { rows: 8, tile: 16 }, 0.90),
            (ProjectionReuseCoordinate { rows: 8, tile: 32 }, 0.70),
        ];
        let anchor = ProjectionReuseCoordinate { rows: 8, tile: 64 };
        let synthesized =
            select_factorized_projection_reuse_coordinate(&candidates, &timings, anchor)
                .expect("factorized projection geometry should synthesize a candidate");

        assert_eq!(synthesized, ProjectionReuseCoordinate { rows: 4, tile: 32 });
        assert!(
            !timings.iter().any(|(measured, _)| *measured == synthesized),
            "rows and tile winners should compose before the interaction verification probe"
        );
    }

    #[test]
    fn forward_projection_cache_key_separates_pair_tail_and_recurrence() {
        let full = ForwardProjectionAutotuneKey {
            device_name: "gpu".to_owned(),
            subgroup_size: 64,
            width: 128,
            head_size: 64,
            batch_pairs: 2,
            has_unpaired_tail: false,
            recurrence: ForwardProjectionRecurrence::Full,
        };
        let packed = ForwardProjectionAutotuneKey {
            recurrence: ForwardProjectionRecurrence::Packed,
            ..full.clone()
        };
        let odd_tail = ForwardProjectionAutotuneKey {
            has_unpaired_tail: true,
            ..full.clone()
        };
        let another_pair = ForwardProjectionAutotuneKey {
            batch_pairs: 3,
            ..full.clone()
        };
        assert_ne!(full, packed);
        assert_ne!(full, odd_tail);
        assert_ne!(full, another_pair);
    }

    #[test]
    fn backward_segment_selector_keeps_deep_current_topology_on_near_tie() {
        let default = BackwardSegmentSchedule {
            state: StateBackwardSchedule::RkvAdd3KeyTransformReduce,
            projection: ProjectionBackwardSchedule::WeightThenFusedMix,
            low_rank_fan_in: Some(RwkvLowRankFanInSchedule::FusedOuter),
        };
        let faster_by_noise = BackwardSegmentSchedule {
            state: StateBackwardSchedule::RkvAdd3,
            projection: ProjectionBackwardSchedule::FusedMixThenWeight,
            low_rank_fan_in: Some(RwkvLowRankFanInSchedule::Split),
        };
        let timings = [(faster_by_noise, 0.991), (default, 1.000)];
        assert_eq!(select_backward_segment_schedule(&timings, default), default);
    }

    #[test]
    fn backward_segment_selector_accepts_material_end_to_end_win() {
        let default = BackwardSegmentSchedule {
            state: StateBackwardSchedule::RkvAdd3KeyTransformReduce,
            projection: ProjectionBackwardSchedule::WeightThenFusedMix,
            low_rank_fan_in: Some(RwkvLowRankFanInSchedule::FusedOuter),
        };
        let winner = BackwardSegmentSchedule {
            state: StateBackwardSchedule::RkvAdd3,
            projection: ProjectionBackwardSchedule::SplitMixThenWeight,
            low_rank_fan_in: Some(RwkvLowRankFanInSchedule::FusedBase),
        };
        let timings = [(winner, 0.900), (default, 1.000)];
        assert_eq!(select_backward_segment_schedule(&timings, default), winner);
    }

    #[test]
    fn backward_segment_cache_separates_core_and_full_cell_profiles() {
        let core = BackwardSegmentAutotuneKey {
            device_name: "gpu".to_owned(),
            subgroup_size: 64,
            width: 128,
            head_size: 64,
            batch: 2,
            w_rank: 0,
            a_rank: 0,
            g_rank: 0,
            full_cell: false,
        };
        let full_cell = BackwardSegmentAutotuneKey {
            w_rank: 32,
            a_rank: 16,
            g_rank: 64,
            full_cell: true,
            ..core.clone()
        };
        assert_ne!(core, full_cell);

        let different_ranks = BackwardSegmentAutotuneKey {
            w_rank: 64,
            ..full_cell.clone()
        };
        assert_ne!(full_cell, different_ranks);
    }

    #[test]
    fn backward_segment_hierarchical_pruning_reduces_three_dimensional_search() {
        let states = [
            StateBackwardSchedule::RkvAdd3,
            StateBackwardSchedule::RkvAdd3KeyTransform,
            StateBackwardSchedule::RkvAdd3KeyTransformReduce,
        ];
        let projections = [
            ProjectionBackwardSchedule::SplitMixThenWeight,
            ProjectionBackwardSchedule::WeightThenSplitMix,
            ProjectionBackwardSchedule::FusedMixThenWeight,
            ProjectionBackwardSchedule::WeightThenFusedMix,
        ];
        let fan_ins = [
            RwkvLowRankFanInSchedule::Split,
            RwkvLowRankFanInSchedule::FusedBase,
            RwkvLowRankFanInSchedule::FusedOuter,
        ];
        let default = BackwardSegmentSchedule {
            state: states[2],
            projection: projections[3],
            low_rank_fan_in: Some(fan_ins[2]),
        };
        let mut timings = Vec::new();
        for (state_rank, &state) in states.iter().enumerate() {
            for (projection_rank, &projection) in projections.iter().enumerate() {
                for (fan_in_rank, &fan_in) in fan_ins.iter().enumerate() {
                    timings.push((
                        BackwardSegmentSchedule {
                            state,
                            projection,
                            low_rank_fan_in: Some(fan_in),
                        },
                        1.0 + state_rank as f64 * 0.10
                            + projection_rank as f64 * 0.02
                            + fan_in_rank as f64 * 0.005,
                    ));
                }
            }
        }

        let finalists = prune_backward_segment_candidates(&timings, default);
        assert_eq!(timings.len(), 36);
        assert_eq!(finalists.len(), 9);
        assert!(finalists.contains(&timings[0].0));
        assert!(finalists.contains(&default));
    }

    #[test]
    fn backward_segment_persistent_cache_roundtrips_rank_aware_key() -> Result<()> {
        let key = BackwardSegmentAutotuneKey {
            device_name: "gpu".to_owned(),
            subgroup_size: 32,
            width: 768,
            head_size: 64,
            batch: 8,
            w_rank: 32,
            a_rank: 16,
            g_rank: 64,
            full_cell: true,
        };
        let schedule = BackwardSegmentSchedule {
            state: StateBackwardSchedule::RkvAdd3KeyTransform,
            projection: ProjectionBackwardSchedule::FusedMixThenWeight,
            low_rank_fan_in: Some(RwkvLowRankFanInSchedule::FusedBase),
        };
        let encoded = serde_json::to_vec(&PersistentBackwardSegmentCache {
            version: BACKWARD_SEGMENT_PERSISTENT_CACHE_VERSION,
            entries: vec![PersistentBackwardSegmentEntry {
                key: key.clone(),
                schedule,
            }],
        })?;
        let decoded: PersistentBackwardSegmentCache = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded.version, BACKWARD_SEGMENT_PERSISTENT_CACHE_VERSION);
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].key, key);
        assert_eq!(decoded.entries[0].schedule, schedule);
        Ok(())
    }

    #[test]
    #[ignore = "GPU projection weight-gradient parity/profile; run explicitly on the target Vulkan device"]
    fn profile_projection_weight_grad_tiled() -> Result<()> {
        let device = VulkanDevice::new()?;
        let width = 128usize;
        let head_size = 64usize;
        let max_batch = 32usize;
        let vector = (0..width)
            .map(|index| ((index * 17 + 5) % 61) as f32 / 97.0 - 0.3)
            .collect::<Vec<_>>();
        let k_k = (0..width)
            .map(|index| 0.5 + ((index * 7 + 3) % 29) as f32 / 113.0)
            .collect::<Vec<_>>();
        let k_a = vec![1.0f32; width];
        let weight = (0..width * width)
            .map(|index| ((index * 37 + 11) % 257) as f32 / 4096.0 - 0.03)
            .collect::<Vec<_>>();
        let op = RwkvTimeMixCoreOp::new(
            device.clone(),
            width,
            head_size,
            max_batch,
            &vector,
            &vector,
            &vector,
            &weight,
            &weight,
            &weight,
            &k_k,
            &k_a,
        )?;
        assert!(
            op.linear3_weight_grad_tiled.is_some(),
            "target Vulkan device exposes no tiled projection dW kernel"
        );

        let weight_len = width * width;
        let weight_groups = [
            super::div_ceil_u32(width, 16),
            super::div_ceil_u32(width, 16),
            1,
        ];
        for batch in [1usize, 2usize, 4usize, 8usize, 16usize, 32usize] {
            let vector_len = batch * width;
            let make_values = |mul: usize, add: usize, modulus: usize, scale: f32, bias: f32| {
                (0..vector_len)
                    .map(|index| ((index * mul + add) % modulus) as f32 / scale + bias)
                    .collect::<Vec<_>>()
            };
            let xr = GpuBuffer::from_f32(&device, &make_values(11, 3, 97, 173.0, -0.21))?;
            let xk = GpuBuffer::from_f32(&device, &make_values(13, 5, 101, 181.0, -0.19))?;
            let xv = GpuBuffer::from_f32(&device, &make_values(17, 7, 103, 191.0, -0.17))?;
            let grad_r = GpuBuffer::from_f32(&device, &make_values(19, 2, 89, 211.0, -0.20))?;
            let grad_k = GpuBuffer::from_f32(&device, &make_values(23, 3, 83, 223.0, -0.17))?;
            let grad_v = GpuBuffer::from_f32(&device, &make_values(29, 5, 79, 227.0, -0.16))?;
            let linear_push = super::LinearPush {
                rows: batch as u32,
                input_dim: width as u32,
                output_dim: width as u32,
            };
            let readback_r = GpuBuffer::zeros_host_f32(&device, weight_len)?;
            let readback_k = GpuBuffer::zeros_host_f32(&device, weight_len)?;
            let readback_v = GpuBuffer::zeros_host_f32(&device, weight_len)?;

            let mut commands = ComputeBatch::new(&device)?;
            op.record_projection_weight_grad_topology_with_inputs(
                &mut commands,
                &xr,
                &xk,
                &xv,
                &grad_r,
                &grad_k,
                &grad_v,
                &linear_push,
                weight_groups,
                ProjectionWeightGradTopology::Baseline,
            )?;
            commands.readback_f32(&op.grad_receptance_weight, &readback_r, weight_len)?;
            commands.readback_f32(&op.grad_key_weight, &readback_k, weight_len)?;
            commands.readback_f32(&op.grad_value_weight, &readback_v, weight_len)?;
            commands.submit()?;
            let baseline_r = readback_r.read_f32(weight_len)?;
            let baseline_k = readback_k.read_f32(weight_len)?;
            let baseline_v = readback_v.read_f32(weight_len)?;

            let mut commands = ComputeBatch::new(&device)?;
            op.record_projection_weight_grad_topology_with_inputs(
                &mut commands,
                &xr,
                &xk,
                &xv,
                &grad_r,
                &grad_k,
                &grad_v,
                &linear_push,
                weight_groups,
                ProjectionWeightGradTopology::Tiled,
            )?;
            commands.readback_f32(&op.grad_receptance_weight, &readback_r, weight_len)?;
            commands.readback_f32(&op.grad_key_weight, &readback_k, weight_len)?;
            commands.readback_f32(&op.grad_value_weight, &readback_v, weight_len)?;
            commands.submit()?;
            assert_eq!(baseline_r, readback_r.read_f32(weight_len)?);
            assert_eq!(baseline_k, readback_k.read_f32(weight_len)?);
            assert_eq!(baseline_v, readback_v.read_f32(weight_len)?);

            let baseline_ms = op.time_projection_weight_grad_topology_ms(
                batch,
                ProjectionWeightGradTopology::Baseline,
            )?;
            let tiled_ms = op.time_projection_weight_grad_topology_ms(
                batch,
                ProjectionWeightGradTopology::Tiled,
            )?;
            println!(
                "projection_weight_grad batch={batch} baseline_ms={baseline_ms:.4} tiled_ms={tiled_ms:.4}"
            );
            let decision = op.choose_projection_weight_grad_topology(batch)?;
            println!(
                "projection_weight_grad batch={batch} selected={} autotuned={}",
                decision.topology.label(),
                decision.autotuned
            );
            assert!(
                op.available_projection_weight_grad_topologies()
                    .contains(&decision.topology),
                "autotuner selected an unavailable projection dW topology"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU forward projection weight-reuse parity/profile; enable the reuse research arm and run explicitly"]
    fn profile_forward_projection_weight_reuse_scaling() -> Result<()> {
        let device = VulkanDevice::new()?;
        let width = 128usize;
        let head_size = 64usize;
        let max_batch = 8usize;
        let vector = (0..width)
            .map(|index| ((index * 17 + 5) % 61) as f32 / 97.0 - 0.3)
            .collect::<Vec<_>>();
        let k_k = (0..width)
            .map(|index| 0.5 + ((index * 7 + 3) % 29) as f32 / 113.0)
            .collect::<Vec<_>>();
        let k_a = vec![1.0f32; width];
        let weight = (0..width * width)
            .map(|index| ((index * 37 + 11) % 257) as f32 / 4096.0 - 0.03)
            .collect::<Vec<_>>();
        let op = RwkvTimeMixCoreOp::new(
            device.clone(),
            width,
            head_size,
            max_batch,
            &vector,
            &vector,
            &vector,
            &weight,
            &weight,
            &weight,
            &k_k,
            &k_a,
        )?;
        assert!(
            op.available_forward_projection_topologies(ForwardProjectionRecurrence::Full, 8)
                .iter()
                .any(|topology| matches!(
                    topology,
                    ForwardProjectionTopology::WeightReuse { rows: 8, .. }
                )),
            "target Vulkan device exposes no reuse8 full-forward topology; set HIERARCHOS_RWKV_TIME_MIX_ENABLE_WEIGHT_REUSE2=1"
        );

        for batch in [2usize, 4usize, 8usize] {
            let vector_len = batch * width;
            let state_len = vector_len * head_size;
            let make_values =
                |len: usize, mul: usize, add: usize, modulus: usize, scale: f32, bias: f32| {
                    (0..len)
                        .map(|index| ((index * mul + add) % modulus) as f32 / scale + bias)
                        .collect::<Vec<_>>()
                };
            let x_norm =
                GpuBuffer::from_f32(&device, &make_values(vector_len, 11, 3, 97, 173.0, -0.21))?;
            let previous =
                GpuBuffer::from_f32(&device, &make_values(vector_len, 13, 5, 101, 181.0, -0.19))?;
            let a =
                GpuBuffer::from_f32(&device, &make_values(vector_len, 17, 7, 103, 191.0, 0.75))?;
            let w = GpuBuffer::from_f32(&device, &make_values(vector_len, 19, 2, 89, 211.0, -2.0))?;
            let state =
                GpuBuffer::from_f32(&device, &make_values(state_len, 23, 3, 83, 1024.0, -0.04))?;
            let mix_push = super::MixPush {
                batch: batch as u32,
                width: width as u32,
            };
            let linear_push = super::LinearPush {
                rows: batch as u32,
                input_dim: width as u32,
                output_dim: width as u32,
            };
            let key_push = super::KeyPush {
                batch: batch as u32,
                width: width as u32,
                head_size: head_size as u32,
            };
            let state_push = super::StatePush {
                batch: batch as u32,
                width: width as u32,
                head_size: head_size as u32,
            };
            let linear_groups = [
                super::div_ceil_u32(width, 16),
                super::div_ceil_u32(batch, 16),
                1,
            ];
            let head_groups = [super::div_ceil_u32(batch * (width / head_size), 64), 1, 1];
            let vector_groups = [super::div_ceil_u32(vector_len, 64), 1, 1];

            let run_topology = |topology: ForwardProjectionTopology| -> Result<_> {
                let xr_readback = GpuBuffer::zeros_host_f32(&device, vector_len)?;
                let xk_readback = GpuBuffer::zeros_host_f32(&device, vector_len)?;
                let xv_readback = GpuBuffer::zeros_host_f32(&device, vector_len)?;
                let r_readback = GpuBuffer::zeros_host_f32(&device, vector_len)?;
                let k_readback = GpuBuffer::zeros_host_f32(&device, vector_len)?;
                let v_readback = GpuBuffer::zeros_host_f32(&device, vector_len)?;
                let state_readback = GpuBuffer::zeros_host_f32(&device, state_len)?;
                let tmix_readback = GpuBuffer::zeros_host_f32(&device, vector_len)?;
                let mut commands = ComputeBatch::new(&device)?;
                op.record_time_mix_projection_key_state_forward_topology(
                    &mut commands,
                    batch,
                    &state,
                    &x_norm,
                    &previous,
                    &a,
                    &w,
                    &mix_push,
                    &linear_push,
                    &key_push,
                    &state_push,
                    linear_groups,
                    head_groups,
                    vector_groups,
                    topology,
                )?;
                commands.readback_f32(&op.xr, &xr_readback, vector_len)?;
                commands.readback_f32(&op.xk, &xk_readback, vector_len)?;
                commands.readback_f32(&op.xv, &xv_readback, vector_len)?;
                commands.readback_f32(&op.r, &r_readback, vector_len)?;
                commands.readback_f32(&op.raw_k, &k_readback, vector_len)?;
                commands.readback_f32(&op.v, &v_readback, vector_len)?;
                commands.readback_f32(&op.new_state, &state_readback, state_len)?;
                commands.readback_f32(&op.tmix, &tmix_readback, vector_len)?;
                commands.submit()?;
                Ok((
                    xr_readback.read_f32(vector_len)?,
                    xk_readback.read_f32(vector_len)?,
                    xv_readback.read_f32(vector_len)?,
                    r_readback.read_f32(vector_len)?,
                    k_readback.read_f32(vector_len)?,
                    v_readback.read_f32(vector_len)?,
                    state_readback.read_f32(state_len)?,
                    tmix_readback.read_f32(vector_len)?,
                ))
            };
            let baseline = run_topology(ForwardProjectionTopology::Baseline)?;
            for topology in op
                .available_forward_projection_topologies(ForwardProjectionRecurrence::Full, batch)
                .into_iter()
                .filter(|topology| *topology != ForwardProjectionTopology::Baseline)
            {
                assert_eq!(baseline, run_topology(topology)?);
                let ms = op.time_full_forward_projection_topology_ms(batch, topology)?;
                println!(
                    "forward_projection batch={batch} topology={} ms={ms:.4}",
                    topology.label()
                );
            }
            let baseline_ms = op.time_full_forward_projection_topology_ms(
                batch,
                ForwardProjectionTopology::Baseline,
            )?;
            let decision = op.choose_full_forward_projection_topology(batch)?;
            println!(
                "forward_projection batch={batch} baseline_ms={baseline_ms:.4} selected={} autotuned={}",
                decision.topology.label(),
                decision.autotuned
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU projection input-gradient parity/profile; run explicitly on the target Vulkan device"]
    fn profile_projection_input_grad_weight_reuse2() -> Result<()> {
        let device = VulkanDevice::new()?;
        let width = 128usize;
        let head_size = 64usize;
        let max_batch = 8usize;
        let vector = (0..width)
            .map(|index| ((index * 17 + 5) % 61) as f32 / 97.0 - 0.3)
            .collect::<Vec<_>>();
        let k_k = (0..width)
            .map(|index| 0.5 + ((index * 7 + 3) % 29) as f32 / 113.0)
            .collect::<Vec<_>>();
        let k_a = vec![1.0f32; width];
        let weight = (0..width * width)
            .map(|index| ((index * 37 + 11) % 257) as f32 / 4096.0 - 0.03)
            .collect::<Vec<_>>();
        let op = RwkvTimeMixCoreOp::new(
            device.clone(),
            width,
            head_size,
            max_batch,
            &vector,
            &vector,
            &vector,
            &weight,
            &weight,
            &weight,
            &k_k,
            &k_a,
        )?;
        assert!(
            op.available_projection_input_grad_topologies(2).len() > 1,
            "target Vulkan device exposes no projection input-gradient reuse tile"
        );

        for batch in [2usize, 3usize, 4usize, 8usize] {
            let vector_len = batch * width;
            let grad_r = (0..vector_len)
                .map(|index| ((index * 13 + 7) % 89) as f32 / 211.0 - 0.2)
                .collect::<Vec<_>>();
            let grad_k = (0..vector_len)
                .map(|index| ((index * 19 + 2) % 83) as f32 / 223.0 - 0.17)
                .collect::<Vec<_>>();
            let grad_v = (0..vector_len)
                .map(|index| ((index * 23 + 3) % 79) as f32 / 227.0 - 0.16)
                .collect::<Vec<_>>();
            let grad_r = GpuBuffer::from_f32(&device, &grad_r)?;
            let grad_k = GpuBuffer::from_f32(&device, &grad_k)?;
            let grad_v = GpuBuffer::from_f32(&device, &grad_v)?;
            let linear_push = super::LinearPush {
                rows: batch as u32,
                input_dim: width as u32,
                output_dim: width as u32,
            };
            let baseline_groups = [
                super::div_ceil_u32(width, 16),
                super::div_ceil_u32(batch, 16),
                1,
            ];

            let readback_r = GpuBuffer::zeros_host_f32(&device, vector_len)?;
            let readback_k = GpuBuffer::zeros_host_f32(&device, vector_len)?;
            let readback_v = GpuBuffer::zeros_host_f32(&device, vector_len)?;
            let mut commands = ComputeBatch::new(&device)?;
            op.record_projection_input_grad_topology(
                &mut commands,
                &grad_r,
                &grad_k,
                &grad_v,
                &linear_push,
                baseline_groups,
                ProjectionInputGradTopology::Baseline,
            )?;
            commands.readback_f32(&op.grad_xr, &readback_r, vector_len)?;
            commands.readback_f32(&op.grad_xk, &readback_k, vector_len)?;
            commands.readback_f32(&op.grad_xv, &readback_v, vector_len)?;
            commands.submit()?;
            let baseline_r = readback_r.read_f32(vector_len)?;
            let baseline_k = readback_k.read_f32(vector_len)?;
            let baseline_v = readback_v.read_f32(vector_len)?;

            for topology in op
                .available_projection_input_grad_topologies(batch)
                .into_iter()
                .filter(|topology| *topology != ProjectionInputGradTopology::Baseline)
            {
                let mut commands = ComputeBatch::new(&device)?;
                op.record_projection_input_grad_topology(
                    &mut commands,
                    &grad_r,
                    &grad_k,
                    &grad_v,
                    &linear_push,
                    baseline_groups,
                    topology,
                )?;
                commands.readback_f32(&op.grad_xr, &readback_r, vector_len)?;
                commands.readback_f32(&op.grad_xk, &readback_k, vector_len)?;
                commands.readback_f32(&op.grad_xv, &readback_v, vector_len)?;
                commands.submit()?;
                assert_eq!(baseline_r, readback_r.read_f32(vector_len)?);
                assert_eq!(baseline_k, readback_k.read_f32(vector_len)?);
                assert_eq!(baseline_v, readback_v.read_f32(vector_len)?);
                let ms = op.time_projection_input_grad_topology_ms(batch, topology)?;
                println!(
                    "projection_input_grad batch={batch} topology={} ms={ms:.4}",
                    topology.label()
                );
            }
            let baseline_ms = op.time_projection_input_grad_topology_ms(
                batch,
                ProjectionInputGradTopology::Baseline,
            )?;
            let decision = op.choose_projection_input_grad_topology(batch)?;
            println!(
                "projection_input_grad batch={batch} baseline_ms={baseline_ms:.4} selected={} autotuned={}",
                decision.topology.label(),
                decision.autotuned
            );
            assert!(
                op.available_projection_input_grad_topologies(batch)
                    .contains(&decision.topology),
                "factorized projection input-gradient autotuner selected an unavailable topology"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU training-backward microprofile; run explicitly on the target Vulkan device"]
    fn profile_state_backward_schedules() -> Result<()> {
        let device = VulkanDevice::new()?;
        let caps = device.subgroup_capabilities();
        let width = 128usize;
        let head_size = 64usize;
        let max_batch = 2usize;
        let vector = vec![0.125f32; width];
        let k_k = vec![0.75f32; width];
        let k_a = vec![1.0f32; width];
        let weight = vec![0.015625f32; width * width];
        let mut op = RwkvTimeMixCoreOp::new(
            device.clone(),
            width,
            head_size,
            max_batch,
            &vector,
            &vector,
            &vector,
            &weight,
            &weight,
            &weight,
            &k_k,
            &k_a,
        )?;

        println!(
            "RWKV state-backward profile device={} subgroup={} width={} head_size={}",
            device.name(),
            caps.subgroup_size,
            width,
            head_size
        );
        for numerics in [
            RwkvNumericsPolicy::StrictParity,
            RwkvNumericsPolicy::FastSubgroup,
            RwkvNumericsPolicy::FastRecurrentTree,
            RwkvNumericsPolicy::FastRecurrentTiled,
            RwkvNumericsPolicy::FastRecurrentSubgroup,
        ] {
            if !op.supports_numerics_policy(numerics) {
                println!("numerics={} unsupported", numerics.label());
                continue;
            }
            op.set_numerics_policy(numerics)?;
            for batch in [1usize, max_batch] {
                for geometry in op.available_backward_kernel_geometries(batch) {
                    op.set_backward_kernel_geometry_label(batch, geometry.label())?;
                    for schedule in op.available_state_backward_schedules(batch) {
                        let ms = op.median_state_backward_schedule_ms(batch, schedule)?;
                        println!(
                            "numerics={} batch={batch} geometry={} schedule={} median_ms={ms:.4}",
                            numerics.label(),
                            geometry.label(),
                            schedule.label()
                        );
                    }
                }
                println!(
                    "numerics={} batch={batch} selected={} autotuned={}",
                    numerics.label(),
                    op.state_backward_schedule_label(batch)
                        .unwrap_or("fallback"),
                    op.state_backward_schedule_was_autotuned(batch)
                );
                for schedule in op.available_backward_segment_schedules(batch, false)? {
                    let ms = op.median_backward_segment_schedule_ms(batch, schedule)?;
                    println!(
                        "numerics={} batch={batch} backward_segment={} median_ms={ms:.4}",
                        numerics.label(),
                        schedule.label()
                    );
                }
                println!(
                    "numerics={} batch={batch} backward_segment_selected={} autotuned={}",
                    numerics.label(),
                    op.backward_segment_schedule_label(batch)
                        .unwrap_or_else(|| "fallback".to_owned()),
                    op.backward_segment_schedule_was_autotuned(batch)
                );
            }
        }
        Ok(())
    }
}
