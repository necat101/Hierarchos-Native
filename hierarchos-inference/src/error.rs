use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("safetensors error: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("missing tensor '{0}'")]
    MissingTensor(String),

    #[error("tensor '{name}' has shape {actual:?}; expected {expected:?}")]
    Shape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("tensor '{name}' must be F32, got {dtype}")]
    Dtype { name: String, dtype: String },

    #[error("unsupported Hierarchos runtime contract: {0}")]
    Unsupported(String),

    #[error("invalid model: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;
