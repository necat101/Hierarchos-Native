//! DeepSeek V4.1 Engram: n-gram hash tables injected into the residual stream
//! before the scheduled blocks.
//!
//! The released implementation derives two conversion artifacts outside the GPU
//! graph: the compressed token map (Unicode normalization of each vocabulary
//! token) and the per-layer NumPy-seeded odd hash multipliers. Both are taken
//! here as validated inputs, so the runtime itself never calls Python or NumPy.
//! Primes, bucket offsets, the persistent compressed-token cache, masked
//! lookback, n-gram hashing, table lookup, and the normalized gate are native.
//!
//! Embedding tables can contain post-dequantization BF16-exact rows or retain
//! packed FP8/UE8M0 bytes. Packed lookup dequantizes only selected rows and
//! rounds them to BF16, matching ParallelEngramEmbedding.forward. Other layers
//! still follow the experimental FP32 graph's explicit precision contract.

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use std::sync::Mutex;

use crate::{vulkan, GpuBuffer, VulkanDevice};

use super::indexer::LinearKernel;
use super::DeepseekV41Checkpoint;
use super::DeepseekV41TextConfig;
use super::{
    DeepseekV41LinearWeights, DeepseekV41MixedLinearKernel, DeepseekV41ResidentLinear,
    DeepseekV41WeightOutput,
};
use super::{DeepseekV41PackedWeights, DeepseekV41ResidentPackedWeights, DeepseekV41WeightFormat};

const CACHE_WRITE: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_engram_cache.spv");
const HASH: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_engram_hash.spv");
const LOOKUP: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_engram_lookup.spv");
const PACKED_LOOKUP: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_engram_packed_lookup.spv");
const GATE: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate.spv");

const MAX_NGRAM: usize = 8;
const DEAD: u32 = u32::MAX;

#[path = "engram_grad.rs"]
mod gradients;
pub use gradients::{
    DeepseekV41EngramAdjoints, DeepseekV41EngramGateConfig, DeepseekV41EngramGateGradients,
    DeepseekV41EngramGateTape, DeepseekV41EngramGradients, DeepseekV41EngramTape,
};

#[cfg(test)]
#[path = "engram_width_tests.rs"]
mod width_tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CachePush {
    rows: u32,
    seq: u32,
    start_pos: u32,
    batch: u32,
    capacity: u32,
    vocab: u32,
    dead: u32,
    has_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HashPush {
    rows: u32,
    seq: u32,
    start_pos: u32,
    batch: u32,
    capacity: u32,
    cols: u32,
    max_ngram: u32,
    n_heads: u32,
    ngram_sizes: u32,
    layer_mult: u32,
    layer_prime: u32,
    pad_id: u32,
    dead: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LookupPush {
    vectors: u32,
    cols: u32,
    head_dim: u32,
    table_rows: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatePush {
    rows: u32,
    hc: u32,
    dim: u32,
    proj_width: u32,
    eps: f32,
    scale: f32,
    has_mask: u32,
}

pub(crate) fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    for prime in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if value % prime == 0 {
            return value == prime;
        }
    }
    let mut d = value - 1;
    let mut shifts = 0u32;
    while d % 2 == 0 {
        d /= 2;
        shifts += 1;
    }
    let mul_mod = |a: u64, b: u64| ((a as u128 * b as u128) % value as u128) as u64;
    let pow_mod = |mut base: u64, mut exponent: u64| {
        let mut result = 1u64;
        base %= value;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = mul_mod(result, base);
            }
            base = mul_mod(base, base);
            exponent >>= 1;
        }
        result
    };
    // Deterministic for every u64.
    for base in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = pow_mod(base % value, d);
        if x == 1 || x == value - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..shifts {
            x = mul_mod(x, x);
            if x == value - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

pub(crate) fn find_next_prime(start: usize, seen: &HashSet<u32>) -> Result<u32> {
    let mut candidate = (start as u64)
        .checked_add(1)
        .context("V4.1 Engram prime search overflow")?;
    loop {
        if candidate > u32::MAX as u64 {
            bail!("V4.1 Engram prime search exceeds u32 buckets");
        }
        if is_prime(candidate) && !seen.contains(&(candidate as u32)) {
            return Ok(candidate as u32);
        }
        candidate += 1;
    }
}

/// Bucket layout of the per-layer n-gram tables, matching released
/// `EngramLayout.from_args`: the primes are drawn in layer, n-gram, head order
/// from `engram_vocab_size - 1` upward, and no prime is reused anywhere.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepseekV41EngramLayout {
    pub max_ngram_size: usize,
    pub layer_ids: Vec<usize>,
    pub num_embeddings: Vec<usize>,
    /// `[layer][ngram_size - 1][head]`
    pub primes: Vec<Vec<Vec<u32>>>,
    /// `[layer][(ngram_size - 1) * n_heads + head]`
    pub offsets: Vec<Vec<u32>>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl DeepseekV41EngramLayout {
    pub fn from_text(text: &DeepseekV41TextConfig) -> Result<Option<Self>> {
        if text.engram_layer_ids.is_empty() {
            return Ok(None);
        }
        let max_ngram_size = text.engram_max_ngram_size;
        let n_heads = text.engram_n_heads;
        if !(2..=MAX_NGRAM).contains(&max_ngram_size)
            || n_heads == 0
            || n_heads > 65_535
            || text.engram_head_dim == 0
        {
            bail!("V4.1 Engram layout dimensions are invalid");
        }
        if text.engram_num_embeddings.len() != text.engram_layer_ids.len()
            || text.engram_vocab_size < 2
        {
            bail!("V4.1 Engram table geometry is invalid");
        }
        let mut unique = HashSet::new();
        for &layer in &text.engram_layer_ids {
            if layer >= text.num_hidden_layers || !unique.insert(layer) {
                bail!("V4.1 Engram layer ids must be unique backbone layers");
            }
        }
        if text
            .engram_num_embeddings
            .iter()
            .any(|&rows| rows == 0 || rows > u32::MAX as usize)
        {
            bail!("V4.1 Engram table row counts are invalid");
        }
        let ngram_sizes = max_ngram_size - 1;
        let mut seen = HashSet::new();
        let mut primes = Vec::with_capacity(text.engram_layer_ids.len());
        let mut offsets = Vec::with_capacity(text.engram_layer_ids.len());
        for _ in &text.engram_layer_ids {
            let mut per_ngram = Vec::with_capacity(ngram_sizes);
            let mut flat = Vec::with_capacity(ngram_sizes * n_heads);
            let mut current = text.engram_vocab_size - 1;
            for _ in 0..ngram_sizes {
                let mut heads = Vec::with_capacity(n_heads);
                for _ in 0..n_heads {
                    let prime = find_next_prime(current, &seen)?;
                    seen.insert(prime);
                    current = prime as usize;
                    heads.push(prime);
                    flat.push(prime);
                }
                per_ngram.push(heads);
            }
            let mut layer_offsets = Vec::with_capacity(flat.len());
            let mut running = 0u64;
            for prime in &flat {
                layer_offsets.push(running as u32);
                running += *prime as u64;
                if running > u32::MAX as u64 {
                    bail!("V4.1 Engram bucket offsets exceed u32");
                }
            }
            primes.push(per_ngram);
            offsets.push(layer_offsets);
        }
        Ok(Some(Self {
            max_ngram_size,
            layer_ids: text.engram_layer_ids.clone(),
            num_embeddings: text.engram_num_embeddings.clone(),
            primes,
            offsets,
            n_heads,
            head_dim: text.engram_head_dim,
        }))
    }

    pub fn n_hash_cols(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_heads
    }

    pub fn layer_hash_index(&self, layer_id: usize) -> Option<usize> {
        self.layer_ids.iter().position(|&id| id == layer_id)
    }
}

/// Conversion artifacts required by the released hash state. `token_map` has one
/// compressed id per tokenizer token; the pooled normalization that produces it
/// lives in the converter. `multipliers` is `[layer][max_ngram_size]` and pins
/// NumPy's seeded values without making the runtime depend on NumPy.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV41EngramTokens {
    pub token_map: Vec<u32>,
    pub multipliers: Vec<Vec<u64>>,
    pub pad_token_id: usize,
}

/// Persistent compressed-token cache plus the per-layer hash kernel.
pub struct DeepseekV41EngramHash {
    device: VulkanDevice,
    layout: DeepseekV41EngramLayout,
    batch: usize,
    capacity: usize,
    vocab: usize,
    pad_id: u32,
    cols: usize,
    cache: GpuBuffer,
    token_map: GpuBuffer,
    multipliers: GpuBuffer,
    primes: GpuBuffer,
    offsets: GpuBuffer,
    cache_kernel: vulkan::ComputeKernel,
    hash_kernel: vulkan::ComputeKernel,
    step: Option<(usize, usize)>,
}

impl DeepseekV41EngramHash {
    pub fn new(
        device: VulkanDevice,
        layout: DeepseekV41EngramLayout,
        tokens: DeepseekV41EngramTokens,
        vocab: usize,
        batch: usize,
        capacity: usize,
    ) -> Result<Self> {
        if batch == 0 || capacity == 0 || vocab == 0 {
            bail!("V4.1 Engram runtime geometry is invalid");
        }
        if tokens.token_map.len() != vocab || tokens.pad_token_id >= vocab {
            bail!("V4.1 Engram token map geometry is invalid");
        }
        let max_token = tokens
            .token_map
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(tokens.token_map[tokens.pad_token_id]);
        if tokens.multipliers.len() != layout.layer_ids.len() {
            bail!("V4.1 Engram multiplier count is invalid");
        }
        let limit = i64::MAX as u128;
        let mut packed = Vec::with_capacity(layout.layer_ids.len() * layout.max_ngram_size * 2);
        for (layer, row) in tokens.multipliers.iter().enumerate() {
            if row.len() != layout.max_ngram_size {
                bail!("V4.1 Engram layer {layer} multiplier count is invalid");
            }
            for &multiplier in row {
                if multiplier == 0 || multiplier % 2 == 0 {
                    bail!("V4.1 Engram hash multipliers must be positive and odd");
                }
                let product = multiplier as u128 * max_token as u128;
                if product >= limit || (multiplier >> 32) * max_token as u64 >= 1u64 << 32 {
                    bail!("V4.1 Engram hash multiplier overflows the 64-bit hash contract");
                }
                packed.push(multiplier as u32);
                packed.push((multiplier >> 32) as u32);
            }
        }
        if packed.len() > u32::MAX as usize {
            bail!("V4.1 Engram multiplier table is too large");
        }
        let mut flat_primes = Vec::new();
        let mut flat_offsets = Vec::new();
        for (layer, per_ngram) in layout.primes.iter().enumerate() {
            if per_ngram.len() != layout.max_ngram_size - 1 {
                bail!("V4.1 Engram layer {layer} prime geometry is invalid");
            }
            for heads in per_ngram {
                if heads.len() != layout.n_heads {
                    bail!("V4.1 Engram layer {layer} prime geometry is invalid");
                }
                for &prime in heads {
                    if prime < 2 || prime >= 1u32 << 31 {
                        bail!("V4.1 Engram primes must fit the 31-bit modulo contract");
                    }
                    flat_primes.push(prime);
                }
            }
            flat_offsets.extend_from_slice(&layout.offsets[layer]);
        }
        if flat_primes.len()
            != layout.layer_ids.len() * layout.n_heads * (layout.max_ngram_size - 1)
            || flat_primes.len() != flat_offsets.len()
            || flat_primes.len() > 65_535
        {
            bail!("V4.1 Engram hash table geometry is invalid");
        }
        let pad_id = tokens.token_map[tokens.pad_token_id];
        let cols = layout.n_hash_cols();
        Ok(Self {
            cache: GpuBuffer::zeros_u32(
                &device,
                batch
                    .checked_mul(capacity)
                    .context("V4.1 Engram cache overflow")?,
            )?,
            token_map: GpuBuffer::from_u32(&device, &tokens.token_map)?,
            multipliers: GpuBuffer::from_u32(&device, &packed)?,
            primes: GpuBuffer::from_u32(&device, &flat_primes)?,
            offsets: GpuBuffer::from_u32(&device, &flat_offsets)?,
            cache_kernel: vulkan::ComputeKernel::new(
                &device,
                CACHE_WRITE,
                4,
                std::mem::size_of::<CachePush>() as u32,
            )?,
            hash_kernel: vulkan::ComputeKernel::new(
                &device,
                HASH,
                5,
                std::mem::size_of::<HashPush>() as u32,
            )?,
            device,
            layout,
            batch,
            capacity,
            vocab,
            pad_id,
            cols,
            step: None,
        })
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Writes this step's compressed token ids into the persistent cache.
    pub fn begin_step(
        &mut self,
        token_ids: &GpuBuffer,
        start_pos: usize,
        seq: usize,
        token_mask: Option<&GpuBuffer>,
    ) -> Result<()> {
        if seq == 0 || start_pos + seq > self.capacity {
            bail!("V4.1 Engram step exceeds the cache");
        }
        let rows = self
            .batch
            .checked_mul(seq)
            .context("V4.1 Engram row overflow")?;
        if rows > 65_535
            || token_ids.f32_capacity() < rows
            || !token_ids.belongs_to_device(&self.device)
        {
            bail!("V4.1 Engram token geometry or device mismatch");
        }
        if let Some(mask) = token_mask {
            if mask.f32_capacity() < rows || !mask.belongs_to_device(&self.device) {
                bail!("V4.1 Engram token mask geometry or device mismatch");
            }
        }
        // The shader reads token ids from a u32 view; the caller uploads them as
        // u32 (the buffer's f32 view is only a capacity check).
        let push = CachePush {
            rows: rows as u32,
            seq: seq as u32,
            start_pos: start_pos as u32,
            batch: self.batch as u32,
            capacity: self.capacity as u32,
            vocab: self.vocab as u32,
            dead: DEAD,
            has_mask: token_mask.is_some() as u32,
        };
        let empty = GpuBuffer::zeros_u32(&self.device, 1)?;
        let mask = token_mask.unwrap_or(&empty);
        let mut commands = self.device.compute_batch()?;
        self.cache_kernel.record_dispatch(
            &mut commands,
            &[token_ids, &self.token_map, &self.cache, mask],
            bytemuck::bytes_of(&push),
            [rows.div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        self.step = Some((start_pos, seq));
        Ok(())
    }

    /// Hash ids `[rows, cols]` for one engram layer, from the cached tokens.
    pub fn hashes(&self, layer_index: usize) -> Result<GpuBuffer> {
        let (start_pos, seq) = self
            .step
            .context("V4.1 Engram hashes require a begun step")?;
        if layer_index >= self.layout.layer_ids.len() {
            bail!("V4.1 Engram layer index is out of range");
        }
        let rows = self.batch * seq;
        let push = HashPush {
            rows: rows as u32,
            seq: seq as u32,
            start_pos: start_pos as u32,
            batch: self.batch as u32,
            capacity: self.capacity as u32,
            cols: self.cols as u32,
            max_ngram: self.layout.max_ngram_size as u32,
            n_heads: self.layout.n_heads as u32,
            ngram_sizes: (self.layout.max_ngram_size - 1) as u32,
            layer_mult: (layer_index * self.layout.max_ngram_size) as u32,
            layer_prime: (layer_index * self.layout.n_heads * (self.layout.max_ngram_size - 1))
                as u32,
            pad_id: self.pad_id,
            dead: DEAD,
        };
        let output = GpuBuffer::zeros_u32(&self.device, rows * self.cols)?;
        let mut commands = self.device.compute_batch()?;
        self.hash_kernel.record_dispatch(
            &mut commands,
            &[
                &self.cache,
                &self.multipliers,
                &self.primes,
                &self.offsets,
                &output,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }
}

pub struct DeepseekV41EngramWeights {
    /// `[num_embeddings, head_dim]`, already dequantized and BF16-rounded.
    pub embedding: Vec<f32>,
    /// `[hidden * (hc_mult + 1), n_hash_cols * head_dim]`
    pub wkv: Vec<f32>,
    /// `[hc_mult, hidden]`
    pub q_weight: Vec<f32>,
    /// `[hc_mult, hidden]`
    pub k_weight: Vec<f32>,
}

enum EngramTable {
    Dense(GpuBuffer),
    Packed(DeepseekV41ResidentPackedWeights),
    /// One table is read and uploaded per lookup, then released.
    Streamed(StreamedEngramTable),
}

/// Checkpoint-backed table source for `EngramTable::Streamed`. The clone shares
/// immutable shard metadata and reads payload bytes only while a lookup runs.
/// Keep package files readable and unchanged for the model's lifetime.
struct StreamedEngramTable {
    checkpoint: Mutex<DeepseekV41Checkpoint>,
    layer_id: usize,
    packed: bool,
}
impl StreamedEngramTable {
    fn stage(
        &self,
        device: &VulkanDevice,
        hashes: &GpuBuffer,
        vectors: usize,
    ) -> Result<(GpuBuffer, usize, StagedEngramTable)> {
        let mut checkpoint = self
            .checkpoint
            .lock()
            .map_err(|_| anyhow::anyhow!("V4.1 streamed Engram checkpoint lock poisoned"))?;
        let (indices, rows) =
            checkpoint.read_engram_rows(self.layer_id, &hashes.read_u32(vectors)?)?;
        let count = rows.shape()[0];
        let table = match rows {
            DeepseekV41LinearWeights::Packed { weight, .. } if self.packed => {
                StagedEngramTable::Packed(weight.upload(device)?)
            }
            DeepseekV41LinearWeights::Fp32 { values, .. } if !self.packed => {
                StagedEngramTable::Dense(GpuBuffer::from_f32(device, &values)?)
            }
            _ => bail!("V4.1 streamed Engram table storage changed after construction"),
        };
        Ok((GpuBuffer::from_u32(device, &indices)?, count, table))
    }
}
enum StagedEngramTable {
    Dense(GpuBuffer),
    Packed(DeepseekV41ResidentPackedWeights),
}
enum EngramProjection {
    Fp32(GpuBuffer),
    Mixed {
        weight: DeepseekV41ResidentLinear,
        kernel: DeepseekV41MixedLinearKernel,
        input_precision: DeepseekV41WeightOutput,
        stream_precision: DeepseekV41WeightOutput,
    },
}

pub struct DeepseekV41Engram {
    device: VulkanDevice,
    layer_hash_index: usize,
    hidden: usize,
    hc: usize,
    head_dim: usize,
    cols: usize,
    table_rows: usize,
    eps: f32,
    scale: f32,
    table: EngramTable,
    linear: LinearKernel,
    wkv: EngramProjection,
    weight: GpuBuffer,
    lookup: vulkan::ComputeKernel,
    lookup_packed: vulkan::ComputeKernel,
    gate: vulkan::ComputeKernel,
    #[cfg(test)]
    trace_projection: std::cell::RefCell<Option<GpuBuffer>>,
}

impl DeepseekV41Engram {
    pub fn new(
        device: VulkanDevice,
        layout: &DeepseekV41EngramLayout,
        layer_hash_index: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        weights: DeepseekV41EngramWeights,
    ) -> Result<Self> {
        Self::new_with_table(
            device,
            layout,
            layer_hash_index,
            hc,
            hidden,
            norm_eps,
            weights,
            None,
            None,
            None,
        )
    }

    /// Keep FP8/UE8M0 rows packed on the device. `weights.embedding` must be
    /// empty; projection/norm parameters follow the existing FP32 contract.
    pub fn new_packed(
        device: VulkanDevice,
        layout: &DeepseekV41EngramLayout,
        layer_hash_index: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        weights: DeepseekV41EngramWeights,
        embedding: DeepseekV41PackedWeights,
    ) -> Result<Self> {
        Self::new_with_table(
            device,
            layout,
            layer_hash_index,
            hc,
            hidden,
            norm_eps,
            weights,
            Some(embedding),
            None,
            None,
        )
    }

    /// Packed table and typed projection. The table emits BF16 lookup values;
    /// quantized projections emit BF16. Stream precision controls the released
    /// residual input/output casts without changing the FP32 gate arithmetic.
    pub fn new_mixed(
        device: VulkanDevice,
        layout: &DeepseekV41EngramLayout,
        layer_hash_index: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        weights: DeepseekV41EngramWeights,
        embedding: DeepseekV41PackedWeights,
        projection: DeepseekV41LinearWeights,
        stream_precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        Self::new_with_table(
            device,
            layout,
            layer_hash_index,
            hc,
            hidden,
            norm_eps,
            weights,
            Some(embedding),
            Some((projection, stream_precision)),
            None,
        )
    }

    /// Dense table read from `checkpoint` for each lookup instead of at
    /// construction. Projection and norms stay resident; only the table is staged.
    pub fn new_streamed(
        device: VulkanDevice,
        layout: &DeepseekV41EngramLayout,
        layer_hash_index: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        weights: DeepseekV41EngramWeights,
        checkpoint: DeepseekV41Checkpoint,
        layer_id: usize,
    ) -> Result<Self> {
        Self::new_with_table(
            device,
            layout,
            layer_hash_index,
            hc,
            hidden,
            norm_eps,
            weights,
            None,
            None,
            Some(StreamedEngramTable {
                checkpoint: Mutex::new(checkpoint),
                layer_id,
                packed: false,
            }),
        )
    }

    /// Packed table staged per lookup with the resident FP32 projection contract.
    pub fn new_packed_streamed(
        device: VulkanDevice,
        layout: &DeepseekV41EngramLayout,
        layer_hash_index: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        weights: DeepseekV41EngramWeights,
        checkpoint: DeepseekV41Checkpoint,
        layer_id: usize,
    ) -> Result<Self> {
        Self::new_with_table(
            device,
            layout,
            layer_hash_index,
            hc,
            hidden,
            norm_eps,
            weights,
            None,
            None,
            Some(StreamedEngramTable {
                checkpoint: Mutex::new(checkpoint),
                layer_id,
                packed: true,
            }),
        )
    }

    /// Packed table staged per lookup plus the typed packed projection.
    #[allow(clippy::too_many_arguments)]
    pub fn new_mixed_streamed(
        device: VulkanDevice,
        layout: &DeepseekV41EngramLayout,
        layer_hash_index: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        weights: DeepseekV41EngramWeights,
        projection: DeepseekV41LinearWeights,
        stream_precision: DeepseekV41WeightOutput,
        checkpoint: DeepseekV41Checkpoint,
        layer_id: usize,
    ) -> Result<Self> {
        Self::new_with_table(
            device,
            layout,
            layer_hash_index,
            hc,
            hidden,
            norm_eps,
            weights,
            None,
            Some((projection, stream_precision)),
            Some(StreamedEngramTable {
                checkpoint: Mutex::new(checkpoint),
                layer_id,
                packed: true,
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_table(
        device: VulkanDevice,
        layout: &DeepseekV41EngramLayout,
        layer_hash_index: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        weights: DeepseekV41EngramWeights,
        packed: Option<DeepseekV41PackedWeights>,
        projection: Option<(DeepseekV41LinearWeights, DeepseekV41WeightOutput)>,
        streamed: Option<StreamedEngramTable>,
    ) -> Result<Self> {
        if layer_hash_index >= layout.layer_ids.len()
            || hc == 0
            || hidden == 0
            || layout.head_dim == 0
        {
            bail!("V4.1 Engram component geometry is invalid");
        }
        let cols = layout.n_hash_cols();
        let head_dim = layout.head_dim;
        let table_rows = layout.num_embeddings[layer_hash_index];
        let input_width = cols
            .checked_mul(head_dim)
            .context("V4.1 Engram projection input overflow")?;
        let output_width = (hc + 1)
            .checked_mul(hidden)
            .context("V4.1 Engram projection output overflow")?;
        if let Some(packed) = &packed {
            if packed.format() != DeepseekV41WeightFormat::EngramFp8Row32
                || packed.rows() != table_rows
                || packed.cols() != head_dim
                || !weights.embedding.is_empty()
                || streamed.is_some()
            {
                bail!("V4.1 packed Engram table format/geometry does not match its layer");
            }
        } else if streamed.is_some() {
            if !weights.embedding.is_empty() {
                bail!("V4.1 streamed Engram table must not be loaded at construction");
            }
        } else if weights.embedding.len() != table_rows * head_dim {
            bail!("V4.1 Engram parameter geometry is invalid");
        }
        if let Some((w, _)) = &projection {
            w.validate()?;
            if w.shape() != [output_width, input_width] || !weights.wkv.is_empty() {
                bail!("V4.1 typed Engram projection geometry is invalid");
            }
        } else if weights.wkv.len() != output_width * input_width {
            bail!("V4.1 Engram parameter geometry is invalid");
        }
        if weights.q_weight.len() != hc * hidden || weights.k_weight.len() != hc * hidden {
            bail!("V4.1 Engram parameter geometry is invalid");
        }
        for (label, values) in [
            ("embedding", &weights.embedding),
            ("wkv", &weights.wkv),
            ("q_weight", &weights.q_weight),
            ("k_weight", &weights.k_weight),
        ] {
            if !values.iter().all(|value| value.is_finite()) {
                bail!("V4.1 Engram {label} must be finite");
            }
        }
        let weight: Vec<f32> = weights
            .q_weight
            .iter()
            .zip(&weights.k_weight)
            .map(|(q, k)| q * k)
            .collect();
        if !norm_eps.is_finite() || norm_eps <= 0.0 {
            bail!("V4.1 Engram norm epsilon is invalid");
        }
        let scale = ((hidden as f64).powf(-0.5)) as f32;
        let table = if let Some(packed) = packed {
            EngramTable::Packed(packed.upload(&device)?)
        } else if let Some(streamed) = streamed {
            EngramTable::Streamed(streamed)
        } else {
            EngramTable::Dense(GpuBuffer::from_f32(&device, &weights.embedding)?)
        };
        let wkv = if let Some((w, stream_precision)) = projection {
            let input_precision = if matches!(w, DeepseekV41LinearWeights::Fp32 { .. }) {
                DeepseekV41WeightOutput::Fp32
            } else {
                DeepseekV41WeightOutput::Bf16
            };
            EngramProjection::Mixed {
                weight: w.upload(&device)?,
                kernel: DeepseekV41MixedLinearKernel::new(device.clone())?,
                input_precision,
                stream_precision,
            }
        } else {
            EngramProjection::Fp32(GpuBuffer::from_f32(&device, &weights.wkv)?)
        };
        Ok(Self {
            table,
            wkv,
            weight: GpuBuffer::from_f32(&device, &weight)?,
            linear: LinearKernel::new(&device)?,
            lookup: vulkan::ComputeKernel::new(
                &device,
                LOOKUP,
                3,
                std::mem::size_of::<LookupPush>() as u32,
            )?,
            lookup_packed: vulkan::ComputeKernel::new(
                &device,
                PACKED_LOOKUP,
                4,
                std::mem::size_of::<LookupPush>() as u32,
            )?,
            gate: vulkan::ComputeKernel::new(
                &device,
                GATE,
                5,
                std::mem::size_of::<GatePush>() as u32,
            )?,
            device,
            layer_hash_index,
            hidden,
            hc,
            head_dim,
            cols,
            table_rows,
            eps: norm_eps,
            scale,
            #[cfg(test)]
            trace_projection: std::cell::RefCell::new(None),
        })
    }

    pub fn layer_hash_index(&self) -> usize {
        self.layer_hash_index
    }
    /// Persistent table payloads; streamed tables are absent between lookups.
    pub(super) fn resident_table_count(&self) -> usize {
        usize::from(!matches!(self.table, EngramTable::Streamed(_)))
    }
    #[cfg(test)]
    pub(super) fn projection_trace(&self) -> Option<GpuBuffer> {
        self.trace_projection.borrow().clone()
    }

    /// `stream [rows, hc, hidden]` plus `hashes [rows, cols]` into the next stream.
    pub fn apply(
        &self,
        stream: &GpuBuffer,
        hashes: &GpuBuffer,
        token_mask: Option<&GpuBuffer>,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let vectors = rows
            .checked_mul(self.cols)
            .context("V4.1 Engram lookup overflow")?;
        if rows == 0
            || vectors > 65_535
            || rows * self.hc > 65_535
            || stream.f32_capacity() < rows * self.hc * self.hidden
            || hashes.f32_capacity() < vectors
            || !stream.belongs_to_device(&self.device)
            || !hashes.belongs_to_device(&self.device)
        {
            bail!("V4.1 Engram input geometry or device mismatch");
        }
        if let Some(mask) = token_mask {
            if mask.f32_capacity() < rows || !mask.belongs_to_device(&self.device) {
                bail!("V4.1 Engram token mask geometry or device mismatch");
            }
        }
        let embedded = self.lookup_rows(hashes, vectors)?;
        let empty = GpuBuffer::zeros_u32(&self.device, 1)?;
        let proj_width = (self.hc + 1) * self.hidden;
        let kv = match &self.wkv {
            EngramProjection::Fp32(w) => self.linear.apply(
                &self.device,
                &embedded,
                w,
                rows,
                self.cols * self.head_dim,
                proj_width,
            )?,
            EngramProjection::Mixed {
                weight,
                kernel,
                input_precision,
                ..
            } => kernel.apply(&embedded, rows, weight, *input_precision)?,
        };
        let stream = match &self.wkv {
            EngramProjection::Mixed {
                kernel,
                stream_precision,
                ..
            } => kernel.cast_buffer(stream, rows * self.hc * self.hidden, *stream_precision)?,
            _ => stream.clone(),
        };
        #[cfg(test)]
        {
            self.trace_projection.replace(Some(kv.clone()));
        }
        let output = GpuBuffer::zeros_f32(&self.device, rows * self.hc * self.hidden)?;
        let gate_push = GatePush {
            rows: rows as u32,
            hc: self.hc as u32,
            dim: self.hidden as u32,
            proj_width: proj_width as u32,
            eps: self.eps,
            scale: self.scale,
            has_mask: token_mask.is_some() as u32,
        };
        let mask = token_mask.unwrap_or(&empty);
        let mut commands = self.device.compute_batch()?;
        self.gate.record_dispatch(
            &mut commands,
            &[&stream, &kv, &self.weight, mask, &output],
            bytemuck::bytes_of(&gate_push),
            [(rows * self.hc) as u32, 1, 1],
        )?;
        commands.submit()?;
        match &self.wkv {
            EngramProjection::Mixed {
                kernel,
                stream_precision,
                ..
            } => kernel.cast_buffer(&output, rows * self.hc * self.hidden, *stream_precision),
            _ => Ok(output),
        }
    }

    pub(super) fn lookup_rows(&self, hashes: &GpuBuffer, vectors: usize) -> Result<GpuBuffer> {
        if vectors == 0
            || vectors > 65_535
            || hashes.f32_capacity() < vectors
            || !hashes.belongs_to_device(&self.device)
        {
            bail!("V4.1 Engram lookup geometry or device mismatch");
        }
        let embedded = GpuBuffer::zeros_f32(&self.device, vectors * self.head_dim)?;
        let mut lookup_push = LookupPush {
            vectors: vectors as u32,
            cols: self.cols as u32,
            head_dim: self.head_dim as u32,
            table_rows: self.table_rows as u32,
        };
        let mut commands = self.device.compute_batch()?;
        // A streamed table is uploaded here, used by this dispatch, and dropped
        // after `submit` completes. Only the lookup result survives the scope.
        let staged;
        let (kernel, bindings) = match &self.table {
            EngramTable::Dense(table) => (&self.lookup, vec![hashes, table, &embedded]),
            EngramTable::Packed(table) => {
                let (data, scales) = table.buffers();
                (&self.lookup_packed, vec![hashes, data, scales, &embedded])
            }
            EngramTable::Streamed(table) => {
                staged = table.stage(&self.device, hashes, vectors)?;
                lookup_push.table_rows = staged.1 as u32;
                match &staged.2 {
                    StagedEngramTable::Dense(table) => {
                        (&self.lookup, vec![&staged.0, table, &embedded])
                    }
                    StagedEngramTable::Packed(table) => {
                        let (data, scales) = table.buffers();
                        (
                            &self.lookup_packed,
                            vec![&staged.0, data, scales, &embedded],
                        )
                    }
                }
            }
        };
        kernel.record_dispatch(
            &mut commands,
            &bindings,
            bytemuck::bytes_of(&lookup_push),
            [vectors as u32, 1, 1],
        )?;
        commands.submit()?;
        drop(bindings);
        Ok(embedded)
    }
}

/// Whether each scheduled Engram table stays resident or is staged per lookup.
/// Projection, norm and gate parameters remain resident under both policies.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepseekV41EngramResidency {
    #[default]
    Resident,
    /// One table is read and uploaded when its Engram layer runs, then released.
    /// No table payload stays resident between layers or calls. Keep checkpoint
    /// files readable and unchanged for the model's lifetime.
    StreamedTables,
}

/// Hash state plus the scheduled per-layer Engram blocks.
pub struct DeepseekV41EngramRuntime {
    precision: DeepseekV41WeightOutput,
    layout: DeepseekV41EngramLayout,
    hash: DeepseekV41EngramHash,
    layers: Vec<Option<DeepseekV41Engram>>,
    token_mask: Option<GpuBuffer>,
}

impl DeepseekV41EngramRuntime {
    pub(super) fn stream_precision(&self) -> DeepseekV41WeightOutput {
        self.precision
    }
    /// Scheduled tables kept resident between calls; staged tables are excluded.
    pub fn resident_engram_table_count(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(DeepseekV41Engram::resident_table_count)
            .sum()
    }
    pub(super) fn validate_text_geometry(
        &self,
        text: &DeepseekV41TextConfig,
        batch: usize,
        capacity: usize,
    ) -> Result<()> {
        let expected = DeepseekV41EngramLayout::from_text(text)?
            .context("V4.1 Engram runtime has no matching configured schedule")?;
        if self.layout != expected
            || self.hash.batch != batch
            || self.hash.capacity < capacity
            || self.hash.vocab != text.vocab_size
            || self.layers.len() != text.num_hidden_layers
        {
            bail!("V4.1 Engram runtime layout/batch/context does not match the backbone");
        }
        for &id in &expected.layer_ids {
            let layer = self.layers[id]
                .as_ref()
                .context("V4.1 Engram layer is missing")?;
            if layer.hidden != text.hidden_size
                || layer.hc != text.hc_mult
                || layer.eps != text.rms_norm_eps as f32
            {
                bail!("V4.1 Engram runtime stream dimensions/epsilon do not match the backbone");
            }
        }
        Ok(())
    }

    /// Token normalization/hash artifacts are non-persistent in the released
    /// checkpoint, so callers must supply them from the matching converter.
    pub fn from_checkpoint(
        device: VulkanDevice,
        checkpoint: &mut super::DeepseekV41Checkpoint,
        tokens: DeepseekV41EngramTokens,
        batch: usize,
        capacity: usize,
    ) -> Result<Self> {
        Self::from_checkpoint_with_precision(
            device,
            checkpoint,
            tokens,
            batch,
            capacity,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    pub fn from_checkpoint_with_precision(
        device: VulkanDevice,
        checkpoint: &mut super::DeepseekV41Checkpoint,
        tokens: DeepseekV41EngramTokens,
        batch: usize,
        capacity: usize,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        Self::from_checkpoint_with_table_residency(
            device,
            checkpoint,
            tokens,
            batch,
            capacity,
            precision,
            DeepseekV41EngramResidency::Resident,
        )
    }

    /// Explicit table residency. `StreamedTables` reads one packed or dense
    /// table per Engram layer application instead of uploading every table at
    /// construction; projection/norm/gate parameters are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn from_checkpoint_with_table_residency(
        device: VulkanDevice,
        checkpoint: &mut super::DeepseekV41Checkpoint,
        tokens: DeepseekV41EngramTokens,
        batch: usize,
        capacity: usize,
        precision: DeepseekV41WeightOutput,
        residency: DeepseekV41EngramResidency,
    ) -> Result<Self> {
        let text = checkpoint.config.text.clone();
        let layout = DeepseekV41EngramLayout::from_text(&text)?
            .context("V4.1 checkpoint has no Engram schedule")?;
        if tokens.pad_token_id != text.engram_pad_token_id
            || tokens
                .token_map
                .iter()
                .any(|&id| id as usize >= text.engram_compressed_vocab_size)
        {
            bail!("V4.1 Engram conversion artifacts do not match configured token bounds/pad id");
        }
        let hash = DeepseekV41EngramHash::new(
            device.clone(),
            layout.clone(),
            tokens,
            text.vocab_size,
            batch,
            capacity,
        )?;
        let mut layers: Vec<Option<DeepseekV41Engram>> =
            (0..text.num_hidden_layers).map(|_| None).collect();
        let streamed = residency == DeepseekV41EngramResidency::StreamedTables;
        for (index, &layer) in layout.layer_ids.iter().enumerate() {
            let packed = checkpoint.tensor_dtype(&format!("layers.{layer}.engram.embed.weight"))?
                == "F8_E4M3";
            let mixed_projection =
                checkpoint.tensor_dtype(&format!("layers.{layer}.engram.wkv.weight"))? != "F32"
                    || precision == DeepseekV41WeightOutput::Bf16
                    || checkpoint.has_lora_target(&format!("layers.{layer}.engram.wkv.weight"));
            layers[layer] = Some(if packed {
                if mixed_projection {
                    if streamed {
                        let (projection, weights) =
                            checkpoint.load_mixed_engram_projection(layer)?;
                        if precision == DeepseekV41WeightOutput::Bf16
                            && matches!(projection, DeepseekV41LinearWeights::Fp32 { .. })
                        {
                            bail!("V4.1 BF16 Engram requires a BF16/packed WKV projection");
                        }
                        DeepseekV41Engram::new_mixed_streamed(
                            device.clone(),
                            &layout,
                            index,
                            text.hc_mult,
                            text.hidden_size,
                            text.rms_norm_eps as f32,
                            weights,
                            projection,
                            precision,
                            checkpoint.clone(),
                            layer,
                        )?
                    } else {
                        let (table, projection, weights) = checkpoint.load_mixed_engram(layer)?;
                        if precision == DeepseekV41WeightOutput::Bf16
                            && matches!(projection, DeepseekV41LinearWeights::Fp32 { .. })
                        {
                            bail!("V4.1 BF16 Engram requires a BF16/packed WKV projection");
                        }
                        DeepseekV41Engram::new_mixed(
                            device.clone(),
                            &layout,
                            index,
                            text.hc_mult,
                            text.hidden_size,
                            text.rms_norm_eps as f32,
                            weights,
                            table,
                            projection,
                            precision,
                        )?
                    }
                } else if streamed {
                    let weights = checkpoint.load_engram_projection(layer)?;
                    DeepseekV41Engram::new_packed_streamed(
                        device.clone(),
                        &layout,
                        index,
                        text.hc_mult,
                        text.hidden_size,
                        text.rms_norm_eps as f32,
                        weights,
                        checkpoint.clone(),
                        layer,
                    )?
                } else {
                    let (table, weights) = checkpoint.load_packed_engram(layer)?;
                    DeepseekV41Engram::new_packed(
                        device.clone(),
                        &layout,
                        index,
                        text.hc_mult,
                        text.hidden_size,
                        text.rms_norm_eps as f32,
                        weights,
                        table,
                    )?
                }
            } else {
                if mixed_projection {
                    let (projection, mut weights) =
                        checkpoint.load_mixed_engram_projection(layer)?;
                    if precision == DeepseekV41WeightOutput::Bf16
                        && matches!(projection, DeepseekV41LinearWeights::Fp32 { .. }) {
                        bail!("V4.1 BF16 Engram requires a BF16/packed WKV projection");
                    }
                    if !streamed {
                        weights.embedding = checkpoint.load_dense_engram_table(layer)?;
                    }
                    DeepseekV41Engram::new_with_table(
                        device.clone(),
                        &layout,
                        index,
                        text.hc_mult,
                        text.hidden_size,
                        text.rms_norm_eps as f32,
                        weights,
                        None,
                        Some((projection, precision)),
                        streamed.then(|| StreamedEngramTable {
                            checkpoint: Mutex::new(checkpoint.clone()),
                            layer_id: layer,
                            packed: false,
                        }),
                    )?
                } else {
                    if precision == DeepseekV41WeightOutput::Bf16 {
                        bail!("V4.1 BF16 Engram checkpoint path requires packed tables and typed projections");
                    }
                    if streamed {
                        let weights = checkpoint.load_engram_projection(layer)?;
                        DeepseekV41Engram::new_streamed(
                            device.clone(),
                            &layout,
                            index,
                            text.hc_mult,
                            text.hidden_size,
                            text.rms_norm_eps as f32,
                            weights,
                            checkpoint.clone(),
                            layer,
                        )?
                    } else {
                        let weights = checkpoint.load_engram(layer)?;
                        DeepseekV41Engram::new(
                            device.clone(),
                            &layout,
                            index,
                            text.hc_mult,
                            text.hidden_size,
                            text.rms_norm_eps as f32,
                            weights,
                        )?
                    }
                }
            });
        }
        Ok(Self {
            layout,
            hash,
            layers,
            token_mask: None,
            precision,
        })
    }

    pub fn new(
        device: VulkanDevice,
        layout: DeepseekV41EngramLayout,
        tokens: DeepseekV41EngramTokens,
        weights: Vec<Option<DeepseekV41EngramWeights>>,
        vocab: usize,
        num_layers: usize,
        hc: usize,
        hidden: usize,
        norm_eps: f32,
        batch: usize,
        capacity: usize,
    ) -> Result<Self> {
        if weights.len() != num_layers {
            bail!("V4.1 Engram layer-weight count is invalid");
        }
        let hash = DeepseekV41EngramHash::new(
            device.clone(),
            layout.clone(),
            tokens,
            vocab,
            batch,
            capacity,
        )?;
        let mut layers = Vec::with_capacity(num_layers);
        for (layer, slot) in weights.into_iter().enumerate() {
            let Some(weights) = slot else {
                if layout.layer_hash_index(layer).is_some() {
                    bail!("V4.1 Engram layer {layer} is missing its parameters");
                }
                layers.push(None);
                continue;
            };
            let index = layout.layer_hash_index(layer).with_context(|| {
                format!("V4.1 Engram weights given for unscheduled layer {layer}")
            })?;
            layers.push(Some(DeepseekV41Engram::new(
                device.clone(),
                &layout,
                index,
                hc,
                hidden,
                norm_eps,
                weights,
            )?));
        }
        Ok(Self {
            layout,
            hash,
            layers,
            token_mask: None,
            precision: DeepseekV41WeightOutput::Fp32,
        })
    }

    pub fn begin_step(
        &mut self,
        token_ids: &GpuBuffer,
        start_pos: usize,
        seq: usize,
        token_mask: Option<&GpuBuffer>,
    ) -> Result<()> {
        self.hash
            .begin_step(token_ids, start_pos, seq, token_mask)?;
        self.token_mask = token_mask.cloned();
        Ok(())
    }

    /// Applies the scheduled Engram block for `layer_id`, if it has one.
    pub fn apply(
        &self,
        layer_id: usize,
        stream: &GpuBuffer,
        rows: usize,
    ) -> Result<Option<GpuBuffer>> {
        let Some(index) = self.layout.layer_hash_index(layer_id) else {
            return Ok(None);
        };
        let hashes = self.hash.hashes(index)?;
        let layer = self.layers[layer_id]
            .as_ref()
            .context("V4.1 Engram parameters are missing")?;
        layer
            .apply(stream, &hashes, self.token_mask.as_ref(), rows)
            .map(Some)
    }

    pub fn layer_ids(&self) -> &[usize] {
        &self.layout.layer_ids
    }

    #[cfg(test)]
    pub(super) fn projected_kv(&self, layer_id: usize) -> Option<GpuBuffer> {
        self.layers
            .get(layer_id)?
            .as_ref()?
            .trace_projection
            .borrow()
            .clone()
    }
}
