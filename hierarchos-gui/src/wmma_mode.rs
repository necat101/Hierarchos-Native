//! WMMA precision choice shown by the Transformer training panel.
//!
//! The desktop control's job is to make the capability boundary visible, not to
//! imply that every listed mode runs. Full FP16 additionally requires the
//! shared package profile and an adapter/driver preflight in NativeApp.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WmmaModeChoice {
    Off,
    Mixed,
    FullFp16,
}

impl WmmaModeChoice {
    pub const ALL: [Self; 3] = [Self::Off, Self::Mixed, Self::FullFp16];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "FP32 (reference)",
            Self::Mixed => "Mixed WMMA + FP32 (AMP-style)",
            Self::FullFp16 => "WMMA Full FP16 (qualified profiles)",
        }
    }

    /// Value forwarded to the CLI's `--wmma-mode`.
    pub fn flag(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Mixed => "mixed",
            Self::FullFp16 => "full-fp16",
        }
    }

    /// Whether this training graph can execute the mode.
    pub fn available_for_training(self) -> bool {
        true
    }

    pub fn unavailable_reason(self) -> &'static str {
        match self {
            Self::Off => {
                "Default. Uses FP32 kernels without requesting hardware matrix acceleration."
            }
            Self::Mixed => {
                "Eligible matrix operations use FP16 operands and FP32 accumulation. Master weights, \
                 gradients, AdamW state, normalization, softmax and loss stay FP32. Other operations \
                 use FP32 kernels. Requires a compatible cooperative-matrix GPU."
            }
            Self::FullFp16 => {
                "Requires a qualified local Llama64 or shared CLIP64/Llama64 package and the \
                 qualified adapter/driver. Eligible reductions up to 64 use genuine FP16 \
                 accumulation, widened on the GPU. Other operations and optimizer state use FP32."
            }
        }
    }

    /// Short line shown under the control.
    pub fn note(self) -> &'static str {
        match self {
            Self::Off => "FP32 reference path. All training operations use FP32.",
            Self::Mixed => {
                "AMP-style mixed precision: eligible matrix operations use FP16 inputs with FP32 \
                 accumulation; master weights and AdamW stay FP32. Other operations stay FP32. \
                 Automatic loss scaling is available for standalone GPT2/Llama full training and \
                 ordinary LoRA and shared CLIP/LLaVA GPT2/Llama image transactions. Other \
                 architecture-specific trainers retain their existing FP32 gradient policy."
            }
            Self::FullFp16 => {
                "Scoped Full FP16 matrix accumulation with FP32 masters and AdamW, GPU loss \
                 scaling and overflow skipping. Local model geometry and adapter/driver must \
                 pass Check backend. Wider reductions retain FP32; other profiles are unavailable."
            }
        }
    }
}

impl Default for WmmaModeChoice {
    fn default() -> Self {
        Self::Off
    }
}
