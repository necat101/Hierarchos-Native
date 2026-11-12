//! WMMA precision choice shown by the Transformer training panel.
//!
//! The desktop control's job is to make the capability boundary visible, not to
//! imply that every listed mode runs. `FullFp16` is listed because it exists as
//! an experimental device capability, but Transformer training refuses it: the
//! graph accumulates in FP16 without an in-graph widening store, so the result
//! could not reach the FP32 masters. The panel renders it disabled and
//! [`WmmaModeChoice::unavailable_reason`] explains why.

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
            Self::FullFp16 => "WMMA Full FP16 (Experimental)",
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
        !matches!(self, Self::FullFp16)
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
                "Unavailable for Transformer training: the graph's GEMMs accumulate in FP16 with no \
                 in-graph widening store, so the result could not reach the FP32 masters. The \
                 primitive-level full-FP16 arm also fails 9 of 37 gates at K >= 64."
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
                 Automatic loss scaling is not implemented."
            }
            Self::FullFp16 => {
                "Full FP16 is experimental and not admitted for Transformer training; it remains a \
                 primitive-level probe only."
            }
        }
    }
}

impl Default for WmmaModeChoice {
    fn default() -> Self {
        Self::Off
    }
}
