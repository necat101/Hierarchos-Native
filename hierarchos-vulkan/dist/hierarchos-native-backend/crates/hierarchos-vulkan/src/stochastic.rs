use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

const CANONICAL_DROPOUT_SPV: &[u8] = include_bytes!("../shaders/canonical_dropout.spv");

pub const HIERARCHOS_CANONICAL_COUNTER_RNG_ALGORITHM: &str = "philox4x32-10-word-v1";

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct HierarchosCanonicalRngState {
    pub algorithm: String,
    pub seed: u64,
    /// Absolute 32-bit output-word cursor. Four consecutive words share one
    /// Philox counter block, but consumers reserve words rather than blocks so
    /// graph topology does not leak into checkpoint semantics.
    pub next_word: u64,
}

impl HierarchosCanonicalRngState {
    pub fn new(seed: u64) -> Self {
        Self {
            algorithm: HIERARCHOS_CANONICAL_COUNTER_RNG_ALGORITHM.to_string(),
            seed,
            next_word: 0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.algorithm != HIERARCHOS_CANONICAL_COUNTER_RNG_ALGORITHM {
            bail!(
                "unsupported canonical stochastic RNG algorithm {:?}",
                self.algorithm
            );
        }
        Ok(())
    }

    /// Reserve an immutable word range for one stochastic graph operation.
    /// Forward rematerialization must reuse the returned reservation instead
    /// of reserving again; this makes checkpointing independent of backend RNG
    /// save/restore behavior.
    pub fn reserve_words(&mut self, word_count: u64) -> Result<HierarchosCanonicalRngReservation> {
        self.validate()?;
        let start_word = self.next_word;
        self.next_word = self
            .next_word
            .checked_add(word_count)
            .context("canonical stochastic RNG word cursor overflow")?;
        Ok(HierarchosCanonicalRngReservation {
            seed: self.seed,
            start_word,
            word_count,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct HierarchosCanonicalRngReservation {
    pub seed: u64,
    pub start_word: u64,
    pub word_count: u64,
}

impl HierarchosCanonicalRngReservation {
    pub fn word(&self, offset: u64) -> Result<u32> {
        if offset >= self.word_count {
            bail!(
                "canonical stochastic RNG reservation has {} words, cannot read offset {}",
                self.word_count,
                offset
            );
        }
        let absolute = self
            .start_word
            .checked_add(offset)
            .context("canonical stochastic RNG reservation offset overflow")?;
        Ok(hierarchos_canonical_philox_word(self.seed, absolute))
    }
}

/// Random123-compatible Philox4x32-10 word mapping used by every backend.
///
/// The logical counter is `word_offset / 4`, stored in counter lanes 0/1; the
/// seed is the 64-bit Philox key. Counter lanes 2/3 are zero in v1. That exact
/// mapping is intentionally part of the ABI and is easy to reproduce in
/// Python/CUDA as well as Vulkan without requiring shader int64 support.
pub fn hierarchos_canonical_philox_word(seed: u64, word_offset: u64) -> u32 {
    let block = word_offset >> 2;
    let lane = (word_offset & 3) as usize;
    let mut counter = [block as u32, (block >> 32) as u32, 0, 0];
    let mut key = [seed as u32, (seed >> 32) as u32];
    for _ in 0..10 {
        let p0 = u64::from(PHILOX_M0) * u64::from(counter[0]);
        let p1 = u64::from(PHILOX_M1) * u64::from(counter[2]);
        counter = [
            (p1 >> 32) as u32 ^ counter[1] ^ key[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ counter[3] ^ key[1],
            p0 as u32,
        ];
        key[0] = key[0].wrapping_add(PHILOX_W0);
        key[1] = key[1].wrapping_add(PHILOX_W1);
    }
    counter[lane]
}

/// Convert a dropout probability to the backend-neutral integer comparison
/// threshold. A sample is dropped exactly when `random_u32 < threshold`.
pub fn hierarchos_dropout_threshold(probability: f64) -> Result<u32> {
    if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
        bail!("dropout probability must be finite and in [0, 1); got {probability}");
    }
    Ok((probability * 4_294_967_296.0).floor() as u32)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CanonicalDropoutPush {
    len: u32,
    seed_lo: u32,
    seed_hi: u32,
    start_word_lo: u32,
    start_word_hi: u32,
    threshold: u32,
    scale: f32,
}

/// Vulkan implementation of the canonical stochastic dropout contract.
///
/// The op owns no RNG cursor. Cursor ownership stays with the training driver;
/// this op consumes an explicit immutable reservation, which is what makes a
/// rematerialized forward replay the exact same mask without saving a Vulkan,
/// CUDA, or PyTorch RNG state blob.
pub struct CanonicalDropoutVulkanOp {
    device: VulkanDevice,
    kernel: vulkan::ComputeKernel,
}

impl CanonicalDropoutVulkanOp {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new_with_access(
                &device,
                CANONICAL_DROPOUT_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<CanonicalDropoutPush>() as u32,
            )?,
            device,
        })
    }

    pub(crate) fn record_apply(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        output: &GpuBuffer,
        reservation: HierarchosCanonicalRngReservation,
        probability: f64,
        len: usize,
    ) -> Result<()> {
        if len == 0 || len > u32::MAX as usize {
            bail!("canonical dropout length must be in 1..=u32::MAX; got {len}");
        }
        if input.f32_capacity() < len || output.f32_capacity() < len {
            bail!("canonical dropout input/output capacity is smaller than length {len}");
        }
        if reservation.word_count < len as u64 {
            bail!(
                "canonical dropout reservation has {} words but needs {len}",
                reservation.word_count
            );
        }
        let threshold = hierarchos_dropout_threshold(probability)?;
        let scale = (1.0 / (1.0 - probability)) as f32;
        if !scale.is_finite() {
            bail!("canonical dropout scale is non-finite for probability {probability}");
        }
        let push = CanonicalDropoutPush {
            len: len as u32,
            seed_lo: reservation.seed as u32,
            seed_hi: (reservation.seed >> 32) as u32,
            start_word_lo: reservation.start_word as u32,
            start_word_hi: (reservation.start_word >> 32) as u32,
            threshold,
            scale,
        };
        self.kernel.record_dispatch(
            commands,
            &[input, output],
            bytemuck::bytes_of(&push),
            [len.div_ceil(256) as u32, 1, 1],
        )
    }

    pub fn apply(
        &self,
        input: &[f32],
        reservation: HierarchosCanonicalRngReservation,
        probability: f64,
    ) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let input_buffer = GpuBuffer::from_f32(&self.device, input)?;
        let output = GpuBuffer::zeros_f32(&self.device, input.len())?;
        let readback = GpuBuffer::zeros_host_f32(&self.device, input.len())?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.record_apply(
            &mut commands,
            &input_buffer,
            &output,
            reservation,
            probability,
            input.len(),
        )?;
        commands.readback_f32(&output, &readback, input.len())?;
        commands.submit()?;
        readback.read_f32(input.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_zero_counter_matches_random123_vector() {
        assert_eq!(
            (0..4)
                .map(|word| hierarchos_canonical_philox_word(0, word))
                .collect::<Vec<_>>(),
            vec![0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn reservations_advance_once_and_replay_by_value() -> Result<()> {
        let mut state = HierarchosCanonicalRngState::new(0x0123_4567_89ab_cdef);
        let first = state.reserve_words(7)?;
        let second = state.reserve_words(5)?;
        assert_eq!(first.start_word, 0);
        assert_eq!(second.start_word, 7);
        assert_eq!(state.next_word, 12);
        assert_eq!(first.word(3)?, first.word(3)?);
        let encoded = serde_json::to_vec(&state)?;
        let decoded: HierarchosCanonicalRngState = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded, state);
        Ok(())
    }

    #[test]
    fn dropout_threshold_has_exact_binary_half_boundary() -> Result<()> {
        assert_eq!(hierarchos_dropout_threshold(0.0)?, 0);
        assert_eq!(hierarchos_dropout_threshold(0.5)?, 0x8000_0000);
        Ok(())
    }

    #[test]
    fn vulkan_dropout_replays_same_reservation_exactly() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let op = CanonicalDropoutVulkanOp::new(device)?;
        let input = [1.0f32, -2.0, 3.5, 4.0, -5.25, 6.0, 7.0, 8.0];
        let mut state = HierarchosCanonicalRngState::new(17);
        let reservation = state.reserve_words(input.len() as u64)?;
        let first = op.apply(&input, reservation, 0.5)?;
        let replay = op.apply(&input, reservation, 0.5)?;
        assert_eq!(first, replay);
        assert_eq!(state.next_word, input.len() as u64);

        let expected = input
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if reservation.word(index as u64).unwrap() < 0x8000_0000 {
                    0.0
                } else {
                    value * 2.0
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(first, expected);
        Ok(())
    }
}
