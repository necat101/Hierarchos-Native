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
            Self::Off => "WMMA Off (FP32)",
            Self::Mixed => "WMMA Mixed (FP16 inputs, FP32 accumulate)",
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
                "Default. Every projection uses the qualified FP32 kernels; no cooperative-matrix \
                 extension is requested from the device."
            }
            Self::Mixed => {
                "Requires VK_KHR_cooperative_matrix with a 16x16x16 f16/f16->f32 subgroup tile. \
                 Distances and logits stay FP32; only GEMM operands narrow to FP16, so logit drift \
                 is bounded by the mixed qualification gate (2e-2 max, 3e-3 RMS)."
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
            Self::Off => "FP32 reference path — the default and the baseline every gate is measured against.",
            Self::Mixed => {
                "Mixed: hardware matrix cores for eligible dense projections; small shapes keep their \
                 tuned FP32 kernels."
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
