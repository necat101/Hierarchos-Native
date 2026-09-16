#![recursion_limit = "512"]

//! Pure-Rust inference for Hierarchos.
//!
//! The first runtime contract deliberately targets the corrected coherent-v9
//! learned function and FP32 weights.  It has no Python, PyTorch, BLAS, CUDA, or
//! C/C++ runtime dependency, which keeps the core suitable for desktop and
//! future Android embedding.

mod bootstrap;
mod error;
mod math;
mod model;
mod rosa;
mod rwkv;
mod sampler;
mod weights;

pub use bootstrap::{initialize_model_package, NativeBootstrapConfig};
pub use error::{Error, Result};
pub use model::{
    HierarchosModel, ModelConfig, RuntimeRwkvStateSnapshot, RuntimeState, RuntimeStateSnapshot,
    RUNTIME_STATE_INTERCHANGE_KIND, RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION,
    RWKV_V8_MATRIX_PACKED_LAYOUT,
};
pub use rosa::{RosaState, RosaStateSnapshot, RosaTransitionSnapshot};
pub use sampler::{Sampler, SamplingConfig};
