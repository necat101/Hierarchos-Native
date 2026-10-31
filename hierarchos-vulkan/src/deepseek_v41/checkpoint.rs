//! Bounded tensor I/O for the released V4.1 checkpoint naming convention.
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

#[path = "checkpoint_cache.rs"]
mod payload_cache;
pub(super) use payload_cache::CacheHandle;
pub use payload_cache::DeepseekV41CheckpointIoStats;

use super::{
    DeepseekV41AttentionLayerWeights, DeepseekV41AttentionWeights, DeepseekV41BackboneLayerWeights,
    DeepseekV41CompressorWeights, DeepseekV41Config, DeepseekV41EngramLayout,
    DeepseekV41EngramWeights, DeepseekV41HyperConnectionWeights, DeepseekV41IndexerWeights,
    DeepseekV41LinearWeights, DeepseekV41MixedAttentionWeights, DeepseekV41MixedExpertWeights,
    DeepseekV41MixedIndexerWeights, DeepseekV41MixedMoeWeights, DeepseekV41MoeWeights,
    DeepseekV41PackedWeights, DeepseekV41WeightFormat,
};

/// Projection classes whose forward goes through `DeepseekV41MixedLinearKernel`.
/// Anything else (grouped `wo_a`, compressor, Engram tables) has a separate
/// kernel and stays fail-closed for adapters.
pub(super) fn lora_target_supported(target: &str) -> bool {
    let Some((module, leaf)) = target.rsplit_once('.') else {
        return false;
    };
    match leaf {
        "wq_a" | "wo_b" => module.ends_with(".attn"),
        "wkv" => module.ends_with(".attn") || module.ends_with(".engram"),
        "wq_b" => module.ends_with(".attn") || module.ends_with(".indexer"),
        "weights_proj" | "wk" => module.ends_with(".indexer"),
        "w1" | "w2" | "w3" => {
            module.ends_with(".ffn.shared_experts") || module.contains(".ffn.experts.")
        }
        _ => false,
    }
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct Shard {
    path: PathBuf,
    data_start: u64,
    file_size: u64,
    tensors: BTreeMap<String, TensorInfo>,
    /// One open handle per shard, reused while the file on disk is unchanged.
    ///
    /// A streamed decode revisits the same shard thousands of times per token
    /// (every routed expert, every layer). Reopening the file per tensor read
    /// paid a `CreateFile`/`CloseHandle` pair for each one; the bytes read are
    /// identical, so the handle is kept and seeked under this lock.
    handle: Mutex<Option<ShardFile>>,
    cache: CacheHandle,
}

struct ShardFile {
    file: File,
    len: u64,
    modified: i64,
}

#[cfg(test)]
mod payload_identity_tests {
    use super::*;
    #[test]
    fn cached_payload_still_rejects_modified_truncated_and_deleted_shards() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "hierarchos-v41-payload-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir_all(&root)?;
        let path = root.join("shard.bin");
        fs::write(&path, [7u8; 16])?;
        let cache = Arc::new(Mutex::new(payload_cache::PayloadCache::default()));
        cache.lock().unwrap().enable();
        let shard = Shard {
            path: path.clone(),
            data_start: 0,
            file_size: 16,
            tensors: BTreeMap::new(),
            handle: Mutex::new(None),
            cache: Arc::clone(&cache),
        };
        assert_eq!(shard.read_at(4, 8)?, vec![7u8; 8]);
        assert_eq!(shard.read_at(4, 8)?, vec![7u8; 8]);
        assert_eq!(cache.lock().unwrap().stats().payload_reads, 1);
        std::thread::sleep(std::time::Duration::from_millis(2));
        fs::write(&path, [9u8; 16])?;
        assert_eq!(shard.read_at(4, 8)?, vec![9u8; 8]);
        assert_eq!(cache.lock().unwrap().stats().invalidations, 1);
        fs::write(&path, [0u8; 3])?;
        assert!(shard.read_at(4, 8).is_err());
        fs::remove_file(&path)?;
        assert!(shard.read_at(4, 8).is_err());
        fs::remove_dir(&root)?;
        Ok(())
    }
}

impl Shard {
    /// Read `length` bytes at `offset`, reusing one handle while the shard file
    /// on disk is still the file that handle was opened from.
    ///
    /// The identity check is what preserves the streamed policies' fail-closed
    /// contract: a deleted, truncated or replaced checkpoint makes the next read
    /// fail exactly as reopening per read did. Only the open/close pair is
    /// avoided, which a streamed decode would otherwise pay thousands of times
    /// per token.
    fn read_at(&self, offset: u64, length: usize) -> Result<Vec<u8>> {
        let metadata = std::fs::metadata(&self.path)
            .with_context(|| format!("reading {} metadata", self.path.display()))?;
        let len = metadata.len();
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|elapsed| elapsed.as_nanos() as i64)
            .unwrap_or(0);
        let mut handle = self
            .handle
            .lock()
            .map_err(|_| anyhow!("V4.1 checkpoint shard handle lock poisoned"))?;
        let reusable = matches!(
            handle.as_ref(),
            Some(file) if file.len == len && file.modified == modified
        );
        if !reusable {
            *handle = Some(ShardFile {
                file: File::open(&self.path)?,
                len,
                modified,
            });
        }
        let file = handle
            .as_mut()
            .expect("checkpoint shard handle was opened or reused");
        let key = payload_cache::Key {
            path: self.path.clone(),
            offset,
            len: length,
            version: (len, modified),
        };
        if let Some(bytes) = self
            .cache
            .lock()
            .map_err(|_| anyhow!("V4.1 payload cache lock poisoned"))?
            .get(&key)
        {
            return Ok(bytes.as_ref().clone());
        }
        file.file.seek(SeekFrom::Start(offset))?;
        let mut raw = vec![0u8; length];
        file.file.read_exact(&mut raw)?;
        self.cache
            .lock()
            .map_err(|_| anyhow!("V4.1 payload cache lock poisoned"))?
            .insert(key, &raw);
        Ok(raw)
    }
}

/// Reads only requested tensor payloads, rather than loading entire model shards.
/// F32/BF16 storage is supported here; compute remains the initial FP32 contract.
/// The separate packed API requires explicit UE8M0 scale names and layouts.
/// The FP32 graph's read_tensor path continues to reject packed storage.
/// Clones share immutable index/header maps and a bounded payload cache when
/// streaming is enabled, and own independent configuration snapshots. File
/// identity is checked even on cache hits. Keep package files unchanged.
#[derive(Clone)]
pub struct DeepseekV41Checkpoint {
    root: PathBuf,
    pub config: DeepseekV41Config,
    weight_map: Option<Arc<BTreeMap<String, String>>>,
    shards: BTreeMap<String, Arc<Shard>>,
    payload_cache: CacheHandle,
    /// Optional PEFT adapter. It is resolved by canonical checkpoint tensor
    /// name, so resident, streamed and test loaders attach it identically.
    lora: Option<Arc<super::peft::DeepseekV41LoraAdapter>>,
}

impl DeepseekV41Checkpoint {
    pub(super) fn enable_payload_cache(&self) -> Result<()> {
        self.payload_cache
            .lock()
            .map_err(|_| anyhow!("V4.1 payload cache lock poisoned"))?
            .enable();
        Ok(())
    }
    pub(super) fn payload_cache_handle(&self) -> CacheHandle {
        Arc::clone(&self.payload_cache)
    }
    pub fn io_stats(&self) -> Result<DeepseekV41CheckpointIoStats> {
        Ok(self
            .payload_cache
            .lock()
            .map_err(|_| anyhow!("V4.1 payload cache lock poisoned"))?
            .stats())
    }
    pub fn tensor_names(&mut self) -> Result<Vec<String>> {
        if let Some(index) = &self.weight_map {
            return Ok(index.keys().cloned().collect());
        }
        Ok(self
            .shard("embed.weight")?
            .tensors
            .keys()
            .cloned()
            .collect())
    }
    /// Every identity-bearing package file, with no tensor payload retained.
    pub fn source_files(&self) -> Result<Vec<PathBuf>> {
        let mut names = std::collections::BTreeSet::new();
        if let Some(index) = &self.weight_map {
            names.extend(index.values().cloned());
        } else {
            names.insert("model.safetensors".to_owned());
        }
        for name in [
            "config.json",
            "model.safetensors.index.json",
            "engram_tokens.json",
            "tokenizer.json",
        ] {
            if self.root.join(name).is_file() {
                names.insert(name.to_owned());
            }
        }
        names
            .into_iter()
            .map(|name| {
                if Path::new(&name)
                    .components()
                    .any(|c| !matches!(c, Component::Normal(_)))
                {
                    bail!("V4.1 identity source is not a plain package-relative path");
                }
                Ok(self.root.join(name))
            })
            .collect()
    }

    /// Gather only requested table rows. Full-table offsets use u64 host I/O;
    /// only the compact selection enters the GPU's u32 address space.
    pub(super) fn read_engram_rows(
        &mut self,
        layer: usize,
        hashes: &[u32],
    ) -> Result<(Vec<u32>, DeepseekV41LinearWeights)> {
        let text = &self.config.text;
        let index = text
            .engram_layer_ids
            .iter()
            .position(|&id| id == layer)
            .context("V4.1 Engram row lookup has no configured layer")?;
        let rows = text.engram_num_embeddings[index];
        let cols = text.engram_head_dim;
        let unique: Vec<u32> = hashes
            .iter()
            .copied()
            .filter(|&id| (id as usize) < rows)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let remapped = hashes
            .iter()
            .map(|id| {
                unique
                    .binary_search(id)
                    .map(|i| i as u32)
                    .unwrap_or(u32::MAX)
            })
            .collect();
        let name = format!("layers.{layer}.engram.embed.weight");
        let dtype = self.tensor_dtype(&name)?;
        let count = unique.len().max(1);
        let row_bytes = cols
            .checked_mul(match dtype.as_str() {
                "F32" => 4,
                "BF16" => 2,
                "F8_E4M3" => 1,
                _ => bail!("V4.1 unsupported Engram table dtype {dtype}"),
            })
            .context("V4.1 Engram row byte count overflow")?;
        let mut data = self.read_rows_payload(&name, [rows, cols], &dtype, row_bytes, &unique)?;
        if unique.is_empty() {
            data.resize(row_bytes, 0);
        }
        let weight = if dtype == "F8_E4M3" {
            if !cols.is_multiple_of(32) {
                bail!("V4.1 packed Engram rows must align to 32");
            }
            let mut scales = self.read_rows_payload(
                &format!("layers.{layer}.engram.embed.scale"),
                [rows, cols / 32],
                "F8_E8M0",
                cols / 32,
                &unique,
            )?;
            if unique.is_empty() {
                scales.resize(cols / 32, 127);
            }
            DeepseekV41LinearWeights::Packed {
                weight: DeepseekV41PackedWeights::new(
                    count,
                    cols,
                    DeepseekV41WeightFormat::EngramFp8Row32,
                    data,
                    scales,
                )?,
                lora: None,
            }
        } else {
            let values: Vec<f32> = if dtype == "F32" {
                crate::weight_offload::f32_values_from_le_bytes(&data)
            } else {
                data.chunks_exact(2)
                    .map(|b| {
                        f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16)
                    })
                    .collect()
            };
            if values.iter().any(|x| !x.is_finite()) {
                bail!("V4.1 non-finite selected Engram row");
            }
            DeepseekV41LinearWeights::Fp32 {
                rows: count,
                cols,
                values,
                lora: None,
            }
        };
        Ok((remapped, weight))
    }

    fn read_rows_payload(
        &mut self,
        name: &str,
        shape: [usize; 2],
        dtype: &str,
        row_bytes: usize,
        indices: &[u32],
    ) -> Result<Vec<u8>> {
        let shard = self.shard(name)?;
        let info = shard
            .tensors
            .get(name)
            .with_context(|| format!("V4.1 shard is missing {name}"))?;
        if info.shape != shape || info.dtype != dtype {
            bail!("V4.1 row source {name} must be {dtype} {shape:?}");
        }
        let bytes = (shape[0] as u64)
            .checked_mul(row_bytes as u64)
            .context("V4.1 row source byte count overflow")?;
        let [begin, end] = info.data_offsets;
        if end.checked_sub(begin) != Some(bytes) || end > shard.file_size - shard.data_start {
            bail!("V4.1 invalid SafeTensors row source offsets for {name}");
        }
        let length = indices
            .len()
            .checked_mul(row_bytes)
            .context("V4.1 row selection overflow")?;
        let mut result = vec![0u8; length];
        for (slot, &row) in indices.iter().enumerate() {
            if row as usize >= shape[0] {
                bail!("V4.1 row selection outside {name}");
            }
            let row_bytes_read = shard.read_at(
                shard.data_start + begin + row as u64 * row_bytes as u64,
                row_bytes,
            )?;
            result[slot * row_bytes..(slot + 1) * row_bytes].copy_from_slice(&row_bytes_read);
        }
        Ok(result)
    }

    #[cfg(test)]
    pub(super) fn shares_metadata_with(&self, other: &Self) -> bool {
        let index = match (&self.weight_map, &other.weight_map) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        index
            && self.shards.len() == other.shards.len()
            && self.shards.iter().all(|(name, shard)| {
                other
                    .shards
                    .get(name)
                    .is_some_and(|other| Arc::ptr_eq(shard, other))
            })
    }

    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let root = directory
            .as_ref()
            .canonicalize()
            .context("opening V4.1 checkpoint directory")?;
        let config = DeepseekV41Config::from_hf_value(&serde_json::from_slice(
            &fs::read(root.join("config.json")).context("reading V4.1 config.json")?,
        )?)?;
        let index = root.join("model.safetensors.index.json");
        let weight_map = if index.is_file() {
            #[derive(Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&fs::read(index)?)?;
            Some(Arc::new(index.weight_map))
        } else {
            None
        };
        Ok(Self {
            root,
            config,
            weight_map,
            shards: BTreeMap::new(),
            payload_cache: Arc::new(Mutex::new(payload_cache::PayloadCache::default())),
            lora: None,
        })
    }

    /// Attach one validated adapter to this package view. The package files are
    /// never modified; clones (including streamed layer sources) share it.
    /// Attach one validated adapter to this package view. The package files are
    /// never modified; clones (including streamed layer sources) share it.
    ///
    /// Every target is resolved against this checkpoint before execution. Only
    /// projections that flow through `read_linear_weight` are accepted: the
    /// grouped `wo_a` output, compressor and Engram tables have their own
    /// kernels and would otherwise accept an adapter that never runs.
    pub fn set_lora_adapter(
        &mut self,
        adapter: Arc<super::peft::DeepseekV41LoraAdapter>,
    ) -> Result<()> {
        for (target, _, _) in adapter.targets() {
            if !lora_target_supported(target) {
                bail!(
                    "V4.1 PEFT target {target:?} is not an additive projection boundary; supported targets are attn wq_a/wq_b/wkv/wo_b, indexer wq_b/weights_proj/wk, and routed/shared expert w1/w2/w3 (wo_a, compressor and Engram tables are fail-closed)"
                );
            }
            let tensor = format!("{target}.weight");
            self.tensor_dtype(&tensor)?;
        }
        self.lora = Some(adapter);
        Ok(())
    }

    pub fn lora_adapter(&self) -> Option<&Arc<super::peft::DeepseekV41LoraAdapter>> {
        self.lora.as_ref()
    }

    fn shard(&mut self, name: &str) -> Result<&Shard> {
        let filename = match &self.weight_map {
            Some(index) => index
                .get(name)
                .with_context(|| format!("V4.1 checkpoint index has no tensor {name}"))?
                .clone(),
            None => "model.safetensors".to_owned(),
        };
        if !self.shards.contains_key(&filename) {
            let relative = Path::new(&filename);
            if relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                bail!(
                    "V4.1 checkpoint shard must be a relative path within the package: {filename}"
                );
            }
            let path = self
                .root
                .join(relative)
                .canonicalize()
                .with_context(|| format!("opening V4.1 shard {filename}"))?;
            if !path.starts_with(&self.root) {
                bail!("V4.1 checkpoint shard resolves outside the package: {filename}");
            }
            let mut file = File::open(&path)?;
            let file_size = file.metadata()?.len();
            let mut length = [0u8; 8];
            file.read_exact(&mut length)
                .context("reading SafeTensors header length")?;
            let header_len = u64::from_le_bytes(length);
            let data_start = header_len
                .checked_add(8)
                .context("SafeTensors header overflow")?;
            if header_len > 64 * 1024 * 1024 || data_start > file_size {
                bail!("V4.1 invalid or oversized SafeTensors header");
            }
            let mut header = vec![0u8; header_len as usize];
            file.read_exact(&mut header)?;
            let mut entries: BTreeMap<String, Value> = serde_json::from_slice(&header)?;
            entries.remove("__metadata__");
            let tensors: BTreeMap<String, TensorInfo> = entries
                .into_iter()
                .map(|(name, value)| Ok((name, serde_json::from_value(value)?)))
                .collect::<Result<_>>()?;
            self.shards.insert(
                filename.clone(),
                Arc::new(Shard {
                    cache: Arc::clone(&self.payload_cache),
                    path,
                    data_start,
                    file_size,
                    tensors,
                    handle: Mutex::new(None),
                }),
            );
        }
        Ok(&self.shards[&filename])
    }

    pub fn tensor_dtype(&mut self, name: &str) -> Result<String> {
        Ok(self
            .shard(name)?
            .tensors
            .get(name)
            .with_context(|| format!("V4.1 shard is missing {name}"))?
            .dtype
            .clone())
    }

    pub fn tensor_shape(&mut self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .shard(name)?
            .tensors
            .get(name)
            .with_context(|| format!("V4.1 shard is missing {name}"))?
            .shape
            .clone())
    }

    pub(super) fn has_lora_target(&self, name: &str) -> bool {
        self.lora
            .as_ref()
            .is_some_and(|a| a.module_for_tensor(name).is_some())
    }

    pub fn read_tensor(&mut self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
        let shard = self.shard(name)?;
        let info = shard
            .tensors
            .get(name)
            .with_context(|| format!("V4.1 shard is missing {name}"))?;
        if info.shape != expected_shape {
            bail!(
                "V4.1 {name} shape {:?}, expected {expected_shape:?}",
                info.shape
            );
        }
        let item_bytes = match info.dtype.as_str() {
            "F32" => 4,
            "BF16" => 2,
            other => bail!("V4.1 tensor {name} uses unsupported storage {other} for the FP32 graph; packed FP8/FP4 requires the explicit packed-weight API and precision policy; use an unquantized F32/BF16 checkpoint"),
        };
        let elements = info
            .shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .context("V4.1 tensor element count overflow")?;
        let bytes = elements
            .checked_mul(item_bytes)
            .context("V4.1 tensor byte count overflow")?;
        let raw = Self::read_payload(shard, name, info, bytes)?;
        let values: Vec<f32> = if item_bytes == 4 {
            crate::weight_offload::f32_values_from_le_bytes(&raw)
        } else {
            raw.chunks_exact(2)
                .map(|word| {
                    f32::from_bits((u16::from_le_bytes(word.try_into().unwrap()) as u32) << 16)
                })
                .collect()
        };
        if values.iter().any(|x| !x.is_finite()) {
            bail!("V4.1 non-finite tensor {name}");
        }
        Ok(values)
    }

    /// Read a BF16 tensor as little-endian 16-bit pairs packed into 32-bit
    /// words. The BF16 vocabulary stream keeps its 129280x5120 tables in this
    /// release format; promoting them to FP32 would double storage past the
    /// device budget without changing any value.
    pub fn read_bf16_words(&mut self, name: &str, expected_shape: &[usize]) -> Result<Vec<u32>> {
        let shard = self.shard(name)?;
        let info = shard
            .tensors
            .get(name)
            .with_context(|| format!("V4.1 shard is missing {name}"))?;
        if info.shape != expected_shape {
            bail!(
                "V4.1 {name} shape {:?}, expected {expected_shape:?}",
                info.shape
            );
        }
        if info.dtype != "BF16" {
            bail!("V4.1 {name} must use BF16 storage for the half vocabulary path");
        }
        let elements = info
            .shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .context("V4.1 tensor element count overflow")?;
        let bytes = elements
            .checked_mul(2)
            .context("V4.1 tensor byte count overflow")?;
        let raw = Self::read_payload(shard, name, info, bytes)?;
        let mut words = Vec::with_capacity(elements.div_ceil(2));
        for pair in raw.chunks_exact(4) {
            words.push(u32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]));
        }
        if elements % 2 == 1 {
            let tail = &raw[raw.len() - 2..];
            words.push(u32::from_le_bytes([tail[0], tail[1], 0, 0]));
        }
        for &word in &words {
            for bits in [word as u16, (word >> 16) as u16] {
                if bits & 0x7f80 == 0x7f80 {
                    bail!("V4.1 non-finite BF16 vocabulary tensor {name}");
                }
            }
        }
        Ok(words)
    }

    /// Read one contiguous run of rows from a BF16 table as packed 16-bit
    /// pairs. The released 129280x5120 vocabulary tables are 1.32 GB each, so a
    /// tiled reader keeps only the rows one operation needs in host memory; the
    /// device then holds a single bounded tile. Element zero of the run is the
    /// start of a table row, so pair packing restarts per run and every value is
    /// the same bits it has in the full-table read.
    pub fn read_bf16_word_range(
        &mut self,
        name: &str,
        shape: [usize; 2],
        first: usize,
        count: usize,
    ) -> Result<Vec<u32>> {
        let raw = self.read_row_run(name, shape, "BF16", first, count)?;
        Self::pack_bf16_words(name, raw, count * shape[1])
    }

    /// Read a contiguous run of rows from an F32 table.
    pub fn read_f32_row_range(
        &mut self,
        name: &str,
        shape: [usize; 2],
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>> {
        let raw = self.read_row_run(name, shape, "F32", first, count)?;
        let values: Vec<f32> = crate::weight_offload::f32_values_from_le_bytes(&raw);
        if values.iter().any(|x| !x.is_finite()) {
            bail!("V4.1 non-finite F32 rows of {name}");
        }
        Ok(values)
    }

    /// Gather a selection of BF16 rows in the order given, packed as 16-bit
    /// pairs across the whole concatenated result. Used for the embedding
    /// lookup, where only the tokens in the current call are needed.
    pub fn read_bf16_rows(
        &mut self,
        name: &str,
        shape: [usize; 2],
        rows: &[u32],
    ) -> Result<Vec<u32>> {
        let row_bytes = shape[1]
            .checked_mul(2)
            .context("V4.1 row byte count overflow")?;
        let raw = self.read_rows_payload(name, shape, "BF16", row_bytes, rows)?;
        Self::pack_bf16_words(name, raw, rows.len() * shape[1])
    }

    /// Gather a selection of F32 rows in the order given.
    pub fn read_f32_rows(
        &mut self,
        name: &str,
        shape: [usize; 2],
        rows: &[u32],
    ) -> Result<Vec<f32>> {
        let row_bytes = shape[1]
            .checked_mul(4)
            .context("V4.1 row byte count overflow")?;
        let raw = self.read_rows_payload(name, shape, "F32", row_bytes, rows)?;
        let values: Vec<f32> = crate::weight_offload::f32_values_from_le_bytes(&raw);
        if values.iter().any(|x| !x.is_finite()) {
            bail!("V4.1 non-finite selected F32 rows of {name}");
        }
        Ok(values)
    }

    fn read_row_run(
        &mut self,
        name: &str,
        shape: [usize; 2],
        dtype: &str,
        first: usize,
        count: usize,
    ) -> Result<Vec<u8>> {
        if count == 0 || first.checked_add(count).is_none_or(|end| end > shape[0]) {
            bail!("V4.1 row run outside {name}");
        }
        let item = match dtype {
            "BF16" => 2usize,
            "F32" => 4,
            other => bail!("V4.1 row run of {name} requires BF16/F32 storage, got {other}"),
        };
        let shard = self.shard(name)?;
        let info = shard
            .tensors
            .get(name)
            .with_context(|| format!("V4.1 shard is missing {name}"))?;
        if info.shape != shape {
            bail!("V4.1 {name} shape {:?}, expected {shape:?}", info.shape);
        }
        if info.dtype != dtype {
            bail!("V4.1 {name} must use {dtype} storage, got {}", info.dtype);
        }
        let row_bytes = shape[1]
            .checked_mul(item)
            .context("V4.1 row byte count overflow")?;
        let total = shape[0]
            .checked_mul(row_bytes)
            .context("V4.1 table byte count overflow")?;
        Self::validate_payload(shard, name, info, total)?;
        let [begin, _] = info.data_offsets;
        shard.read_at(
            shard.data_start + begin + first as u64 * row_bytes as u64,
            count * row_bytes,
        )
    }

    fn pack_bf16_words(name: &str, raw: Vec<u8>, elements: usize) -> Result<Vec<u32>> {
        let mut words = Vec::with_capacity(elements.div_ceil(2));
        for pair in raw.chunks_exact(4) {
            words.push(u32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]));
        }
        if elements % 2 == 1 {
            let tail = &raw[raw.len() - 2..];
            words.push(u32::from_le_bytes([tail[0], tail[1], 0, 0]));
        }
        for &word in &words {
            for bits in [word as u16, (word >> 16) as u16] {
                if bits & 0x7f80 == 0x7f80 {
                    bail!("V4.1 non-finite BF16 vocabulary tensor {name}");
                }
            }
        }
        Ok(words)
    }

    /// Header-only vocabulary preflight; payload finiteness is checked when
    /// each tensor is read for its lookup/projection.
    pub(super) fn validate_vocabulary_metadata(&mut self, half: bool) -> Result<()> {
        let shape = [self.config.text.vocab_size, self.config.text.hidden_size];
        for name in ["embed.weight", "head.weight"] {
            let shard = self.shard(name)?;
            let info = shard
                .tensors
                .get(name)
                .with_context(|| format!("V4.1 shard is missing {name}"))?;
            if info.shape != shape {
                bail!("V4.1 {name} shape {:?}, expected {shape:?}", info.shape);
            }
            if half && info.dtype != "BF16" {
                bail!("V4.1 {name} must use BF16 storage for the half vocabulary path");
            }
            let item_bytes = match info.dtype.as_str() {
                "F32" => 4,
                "BF16" => 2,
                _ => bail!("V4.1 vocabulary tensor {name} requires F32/BF16 storage"),
            };
            let bytes = shape
                .iter()
                .try_fold(item_bytes, |n: usize, &dim| n.checked_mul(dim))
                .context("V4.1 vocabulary byte count overflow")?;
            Self::validate_payload(shard, name, info, bytes)?;
        }
        Ok(())
    }

    fn validate_payload(shard: &Shard, name: &str, info: &TensorInfo, bytes: usize) -> Result<()> {
        let [begin, end] = info.data_offsets;
        if end.checked_sub(begin) != Some(bytes as u64) || end > shard.file_size - shard.data_start
        {
            bail!("V4.1 invalid SafeTensors offsets for {name}");
        }
        Ok(())
    }

    fn read_payload(shard: &Shard, name: &str, info: &TensorInfo, bytes: usize) -> Result<Vec<u8>> {
        Self::validate_payload(shard, name, info, bytes)?;
        let [begin, _] = info.data_offsets;
        shard.read_at(shard.data_start + begin, bytes)
    }

    /// Read only the named packed weight and scale payloads. Names are explicit:
    /// exported HF tensors use weight_scale_inv, converted tensors use scale.
    /// F4 headers describe logical [N,K], whereas exported I8 stores [N,K/2].
    /// This does not promote packed weights into the FP32 model's linear path.
    pub fn read_packed_weight(
        &mut self,
        name: &str,
        scale_name: &str,
        rows: usize,
        cols: usize,
        format: DeepseekV41WeightFormat,
    ) -> Result<DeepseekV41PackedWeights> {
        self.validate_packed_weight(name, scale_name, rows, cols, format)?;
        let elements = rows * cols;
        let shard = self.shard(name)?;
        let info = &shard.tensors[name];
        let data = Self::read_payload(
            shard,
            name,
            info,
            if format == DeepseekV41WeightFormat::Fp4Row32 {
                elements / 2
            } else {
                elements
            },
        )?;
        let shard = self.shard(scale_name)?;
        let info = &shard.tensors[scale_name];
        let shape = format.scale_shape(rows, cols);
        let scales = Self::read_payload(shard, scale_name, info, shape[0] * shape[1])?;
        DeepseekV41PackedWeights::new(rows, cols, format, data, scales)
    }

    /// Validate complete tensor/scale metadata without allocating payloads.
    fn validate_packed_weight(
        &mut self,
        name: &str,
        scale_name: &str,
        rows: usize,
        cols: usize,
        format: DeepseekV41WeightFormat,
    ) -> Result<()> {
        let elements = rows
            .checked_mul(cols)
            .context("V4.1 packed weight count overflow")?;
        if rows == 0
            || cols == 0
            || !cols.is_multiple_of(format.block_size())
            || elements > u32::MAX as usize
        {
            bail!("V4.1 invalid packed checkpoint geometry");
        }
        let shard = self.shard(name)?;
        let info = shard
            .tensors
            .get(name)
            .with_context(|| format!("V4.1 shard is missing {name}"))?;
        let fp4 = format == DeepseekV41WeightFormat::Fp4Row32;
        let expected_shape = if fp4 && info.dtype == "I8" {
            [rows, cols / 2]
        } else {
            [rows, cols]
        };
        if info.shape != expected_shape
            || if fp4 {
                !matches!(info.dtype.as_str(), "F4" | "I8")
            } else {
                info.dtype != "F8_E4M3"
            }
        {
            bail!("V4.1 packed weight {name} dtype/shape does not match {format:?} logical [{rows},{cols}]");
        }
        Self::validate_payload(shard, name, info, if fp4 { elements / 2 } else { elements })?;
        let shard = self.shard(scale_name)?;
        let info = shard
            .tensors
            .get(scale_name)
            .with_context(|| format!("V4.1 shard is missing {scale_name}"))?;
        let shape = format.scale_shape(rows, cols);
        if info.dtype != "F8_E8M0" || info.shape != shape {
            bail!("V4.1 scale {scale_name} must be F8_E8M0 with shape {shape:?}");
        }
        Self::validate_payload(shard, scale_name, info, shape[0] * shape[1])
    }

    /// Converted inference names. Storage precision is preserved; scales for
    /// packed projections must use the module's exact `.scale` tensor name.
    /// The caller specifies the FP8 layout; FP4 experts have the released row32 layout.
    pub fn read_linear_weight(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        fp8_format: DeepseekV41WeightFormat,
    ) -> Result<DeepseekV41LinearWeights> {
        let dtype = self.tensor_dtype(name)?;
        let mut weights: DeepseekV41LinearWeights = match dtype.as_str() {
            "F32" => DeepseekV41LinearWeights::Fp32 {
                rows,
                cols,
                values: self.read_tensor(name, &[rows, cols])?,
                lora: None,
            },
            "BF16" => DeepseekV41LinearWeights::Bf16 {
                rows,
                cols,
                values: self.read_tensor(name, &[rows, cols])?,
                lora: None,
            },
            "F8_E4M3" | "F4" | "I8" => {
                let prefix = name
                    .strip_suffix(".weight")
                    .context("V4.1 packed linear name must end with .weight")?;
                let format = if dtype == "F8_E4M3" {
                    if !matches!(
                        fp8_format,
                        DeepseekV41WeightFormat::Fp8Block32 | DeepseekV41WeightFormat::Fp8Block128
                    ) {
                        bail!("V4.1 packed linear requires an explicit FP8 block layout");
                    }
                    fp8_format
                } else {
                    DeepseekV41WeightFormat::Fp4Row32
                };
                DeepseekV41LinearWeights::Packed {
                    weight: self.read_packed_weight(
                        name,
                        &format!("{prefix}.scale"),
                        rows,
                        cols,
                        format,
                    )?,
                    lora: None,
                }
            }
            _ => bail!("V4.1 unsupported linear weight dtype {dtype}"),
        };
        if let Some(module) = self
            .lora
            .as_ref()
            .and_then(|adapter| adapter.module_for_tensor(name))
        {
            weights.attach_lora(module)?;
        }
        Ok(weights)
    }

    pub fn load_mixed_moe(&mut self, layer_id: usize) -> Result<DeepseekV41MixedMoeWeights> {
        let text = self.config.text.clone();
        if layer_id >= text.num_hidden_layers {
            bail!("V4.1 MoE layer is out of range");
        }
        let prefix = format!("layers.{layer_id}.ffn");
        let hidden = text.hidden_size;
        let experts = (0..text.n_routed_experts)
            .map(|id| self.load_mixed_expert(layer_id, id))
            .collect::<Result<_>>()?;
        let shared = self.read_mixed_expert(&format!("{prefix}.shared_experts"))?;
        Ok(DeepseekV41MixedMoeWeights {
            gate_weight: self.read_tensor(
                &format!("{prefix}.gate.weight"),
                &[text.n_routed_experts, hidden],
            )?,
            gate_bias: self
                .read_tensor(&format!("{prefix}.gate.bias"), &[text.n_routed_experts])?,
            experts,
            shared,
        })
    }

    fn read_mixed_expert(&mut self, name: &str) -> Result<DeepseekV41MixedExpertWeights> {
        let (hidden, inter) = (
            self.config.text.hidden_size,
            self.config.text.moe_intermediate_size,
        );
        Ok(DeepseekV41MixedExpertWeights {
            w1: self.read_linear_weight(
                &format!("{name}.w1.weight"),
                inter,
                hidden,
                DeepseekV41WeightFormat::Fp8Block32,
            )?,
            w2: self.read_linear_weight(
                &format!("{name}.w2.weight"),
                hidden,
                inter,
                DeepseekV41WeightFormat::Fp8Block32,
            )?,
            w3: self.read_linear_weight(
                &format!("{name}.w3.weight"),
                inter,
                hidden,
                DeepseekV41WeightFormat::Fp8Block32,
            )?,
        })
    }

    pub(super) fn load_mixed_expert(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<DeepseekV41MixedExpertWeights> {
        if layer >= self.config.text.num_hidden_layers
            || expert >= self.config.text.n_routed_experts
        {
            bail!("V4.1 streamed expert is out of range");
        }
        self.read_mixed_expert(&format!("layers.{layer}.ffn.experts.{expert}"))
    }

    pub(super) fn load_streamed_moe(
        &mut self,
        layer: usize,
        stream_shared: bool,
    ) -> Result<super::backbone::StreamedMoeWeights> {
        self.enable_payload_cache()?;
        let text = self.config.text.clone();
        if layer >= text.num_hidden_layers || !(1..=512).contains(&text.n_routed_experts) {
            bail!("V4.1 streamed MoE layer or expert count is invalid");
        }
        // Check every expert before inference, including experts not selected by
        // a particular prompt. Preflight itself retains only headers/index.
        for expert in 0..text.n_routed_experts {
            self.validate_streamed_expert(
                &format!("layers.{layer}.ffn.experts.{expert}"),
                "routed",
            )?;
        }
        let prefix = format!("layers.{layer}.ffn");
        let shared = if stream_shared {
            self.validate_streamed_expert(&format!("{prefix}.shared_experts"), "shared")?;
            None
        } else {
            Some(self.load_mixed_shared_expert(layer)?)
        };
        Ok(super::backbone::StreamedMoeWeights {
            gate_weight: self.read_tensor(
                &format!("{prefix}.gate.weight"),
                &[text.n_routed_experts, text.hidden_size],
            )?,
            gate_bias: self
                .read_tensor(&format!("{prefix}.gate.bias"), &[text.n_routed_experts])?,
            shared,
            checkpoint: self.clone(),
            layer,
        })
    }

    pub(super) fn load_mixed_shared_expert(
        &mut self,
        layer: usize,
    ) -> Result<DeepseekV41MixedExpertWeights> {
        if layer >= self.config.text.num_hidden_layers {
            bail!("V4.1 shared expert layer is out of range");
        }
        self.read_mixed_expert(&format!("layers.{layer}.ffn.shared_experts"))
    }

    fn validate_streamed_expert(&mut self, name: &str, kind: &str) -> Result<()> {
        let (hidden, inter) = (
            self.config.text.hidden_size,
            self.config.text.moe_intermediate_size,
        );
        for (module, rows, cols) in [
            ("w1", inter, hidden),
            ("w2", hidden, inter),
            ("w3", inter, hidden),
        ] {
            let prefix = format!("{name}.{module}");
            let weight = format!("{prefix}.weight");
            let dtype = self.tensor_dtype(&weight)?;
            if matches!(dtype.as_str(), "F32" | "BF16") {
                if self.tensor_shape(&weight)? != [rows, cols] {
                    bail!("V4.1 streamed {kind} expert {weight} shape mismatch");
                }
                let stream = self.tensor_dtype("embed.weight")?;
                if dtype != stream {
                    bail!("V4.1 streamed {kind} expert {weight} storage {dtype} does not match {stream} stream");
                }
                continue;
            }
            let format = match dtype.as_str() {
                "F8_E4M3" => DeepseekV41WeightFormat::Fp8Block32,
                "F4" | "I8" => DeepseekV41WeightFormat::Fp4Row32,
                _ => bail!("V4.1 streamed MoE requires packed {kind} experts: {weight}"),
            };
            self.validate_packed_weight(&weight, &format!("{prefix}.scale"), rows, cols, format)?;
        }
        Ok(())
    }

    /// Load the released attention tensor names, including only the compressor
    /// and indexer tensors owned by this backbone layer. Shared consumers have
    /// no duplicate wk/k_norm or compressor payloads in the checkpoint.
    pub fn load_attention_layer(
        &mut self,
        layer_id: usize,
    ) -> Result<DeepseekV41AttentionLayerWeights> {
        let plan = self.config.plan()?;
        let layer = plan
            .backbone
            .get(layer_id)
            .context("V4.1 attention layer is out of range")?;
        let text = self.config.text.clone();
        let prefix = format!("layers.{layer_id}.attn");
        let attention = DeepseekV41AttentionWeights {
            wq_a: self.read_tensor(
                &format!("{prefix}.wq_a.weight"),
                &[text.q_lora_rank, text.hidden_size],
            )?,
            q_norm: self.read_tensor(&format!("{prefix}.q_norm.weight"), &[text.q_lora_rank])?,
            wq_b: self.read_tensor(
                &format!("{prefix}.wq_b.weight"),
                &[text.num_attention_heads * text.head_dim, text.q_lora_rank],
            )?,
            wkv: self.read_tensor(
                &format!("{prefix}.wkv.weight"),
                &[text.head_dim, text.hidden_size],
            )?,
            kv_norm: self.read_tensor(&format!("{prefix}.kv_norm.weight"), &[text.head_dim])?,
            wo_a: self.read_tensor(
                &format!("{prefix}.wo_a.weight"),
                &[
                    text.o_groups * text.o_lora_rank,
                    text.num_attention_heads / text.o_groups * text.head_dim,
                ],
            )?,
            wo_b: self.read_tensor(
                &format!("{prefix}.wo_b.weight"),
                &[text.hidden_size, text.o_groups * text.o_lora_rank],
            )?,
            attn_sink: self
                .read_tensor(&format!("{prefix}.attn_sink"), &[text.num_attention_heads])?,
        };
        let owns_k = layer.kv_source_layer == Some(layer_id);
        let compressor = if owns_k {
            Some(self.load_compressor(layer_id)?)
        } else {
            None
        };
        let indexer = if layer.index_source_layer == Some(layer_id) {
            Some(DeepseekV41IndexerWeights {
                wk: if owns_k {
                    self.read_tensor(
                        &format!("{prefix}.indexer.wk.weight"),
                        &[text.index_head_dim, text.head_dim],
                    )?
                } else {
                    Vec::new()
                },
                k_norm: if owns_k {
                    self.read_tensor(
                        &format!("{prefix}.indexer.k_norm.weight"),
                        &[text.index_head_dim],
                    )?
                } else {
                    Vec::new()
                },
                wq_b: self.read_tensor(
                    &format!("{prefix}.indexer.wq_b.weight"),
                    &[text.index_n_heads * text.index_head_dim, text.q_lora_rank],
                )?,
                weights_proj: self.read_tensor(
                    &format!("{prefix}.indexer.weights_proj.weight"),
                    &[text.index_n_heads, text.hidden_size],
                )?,
            })
        } else {
            None
        };
        Ok(DeepseekV41AttentionLayerWeights {
            attention,
            compressor,
            indexer,
        })
    }

    /// Preserve projection storage while retaining the existing F32 layer API.
    /// Only owned indexer/key payloads are read. Packed tensors stay packed.
    pub(super) fn load_attention_components(
        &mut self,
        layer_id: usize,
    ) -> Result<super::orchestrator::AttentionLayerWeights> {
        use super::orchestrator::{AttentionLayerWeights, AttentionWeights, IndexerWeights};
        let plan = self.config.plan()?;
        let layer = plan
            .backbone
            .get(layer_id)
            .context("V4.1 attention layer is out of range")?;
        let text = self.config.text.clone();
        let prefix = format!("layers.{layer_id}.attn");
        let owns_k = layer.kv_source_layer == Some(layer_id);
        let owns_index = layer.index_source_layer == Some(layer_id);
        let mut names = vec!["wq_a", "wq_b", "wkv", "wo_a", "wo_b"];
        if owns_index {
            names.extend(["indexer.wq_b", "indexer.weights_proj"]);
            if owns_k {
                names.push("indexer.wk");
            }
        }
        let mut typed = self.lora.is_some();
        for name in names {
            typed |= self.tensor_dtype(&format!("{prefix}.{name}.weight"))? != "F32";
        }
        if !typed {
            return Ok(self.load_attention_layer(layer_id)?.into());
        }
        let mut linear = |name: &str, rows, cols| {
            self.read_linear_weight(
                &format!("{prefix}.{name}.weight"),
                rows,
                cols,
                DeepseekV41WeightFormat::Fp8Block32,
            )
        };
        let wq_a = linear("wq_a", text.q_lora_rank, text.hidden_size)?;
        let wq_b = linear(
            "wq_b",
            text.num_attention_heads * text.head_dim,
            text.q_lora_rank,
        )?;
        let wkv = linear("wkv", text.head_dim, text.hidden_size)?;
        let wo_a = linear(
            "wo_a",
            text.o_groups * text.o_lora_rank,
            text.num_attention_heads / text.o_groups * text.head_dim,
        )?;
        let wo_b = linear("wo_b", text.hidden_size, text.o_groups * text.o_lora_rank)?;
        let index_projections = if owns_index {
            Some((
                linear(
                    "indexer.wq_b",
                    text.index_n_heads * text.index_head_dim,
                    text.q_lora_rank,
                )?,
                linear("indexer.weights_proj", text.index_n_heads, text.hidden_size)?,
                if owns_k {
                    Some(linear("indexer.wk", text.index_head_dim, text.head_dim)?)
                } else {
                    None
                },
            ))
        } else {
            None
        };
        let indexer = index_projections
            .map(|(wq_b, weights_proj, wk)| -> Result<_> {
                Ok(IndexerWeights::Mixed(DeepseekV41MixedIndexerWeights {
                    wq_b,
                    weights_proj,
                    wk,
                    k_norm: if owns_k {
                        self.read_tensor(
                            &format!("{prefix}.indexer.k_norm.weight"),
                            &[text.index_head_dim],
                        )?
                    } else {
                        vec![]
                    },
                }))
            })
            .transpose()?;
        Ok(AttentionLayerWeights {
            attention: AttentionWeights::Mixed(DeepseekV41MixedAttentionWeights {
                wq_a,
                wq_b,
                wkv,
                wo_a,
                wo_b,
                q_norm: self
                    .read_tensor(&format!("{prefix}.q_norm.weight"), &[text.q_lora_rank])?,
                kv_norm: self.read_tensor(&format!("{prefix}.kv_norm.weight"), &[text.head_dim])?,
                attn_sink: self
                    .read_tensor(&format!("{prefix}.attn_sink"), &[text.num_attention_heads])?,
            }),
            compressor: if owns_k {
                Some(self.load_compressor(layer_id)?)
            } else {
                None
            },
            indexer,
        })
    }

    pub fn load_backbone_layer(
        &mut self,
        layer_id: usize,
    ) -> Result<DeepseekV41BackboneLayerWeights> {
        let w = self.load_backbone_components(layer_id)?;
        let super::backbone::MoeLayerWeights::Fp32(moe) = w.moe else {
            bail!("V4.1 packed MoE storage requires the mixed backbone checkpoint constructor");
        };
        Ok(DeepseekV41BackboneLayerWeights {
            attention: w
                .attention
                .context("V4.1 resident backbone layer has no attention weights")?
                .into_fp32()?,
            hyper: w.hyper,
            moe,
            attn_norm: w.attn_norm,
            ffn_norm: w.ffn_norm,
        })
    }

    pub(super) fn load_backbone_components(
        &mut self,
        layer_id: usize,
    ) -> Result<super::backbone::BackboneLayerWeights> {
        self.load_backbone_components_with_residency(
            layer_id,
            super::DeepseekV41ExpertResidency::Resident,
        )
    }

    pub(super) fn load_backbone_components_with_residency(
        &mut self,
        layer_id: usize,
        residency: super::DeepseekV41ExpertResidency,
    ) -> Result<super::backbone::BackboneLayerWeights> {
        let attention = Some(self.load_attention_components(layer_id)?);
        self.load_backbone_rest(layer_id, residency, attention)
    }

    /// Backbone layer weights without attention/indexer/compressor payloads.
    /// The streamed orchestrator reads those per layer instead.
    pub(super) fn load_backbone_components_without_attention(
        &mut self,
        layer_id: usize,
        residency: super::DeepseekV41ExpertResidency,
    ) -> Result<super::backbone::BackboneLayerWeights> {
        self.load_backbone_rest(layer_id, residency, None)
    }

    fn load_backbone_rest(
        &mut self,
        layer_id: usize,
        residency: super::DeepseekV41ExpertResidency,
        attention: Option<super::orchestrator::AttentionLayerWeights>,
    ) -> Result<super::backbone::BackboneLayerWeights> {
        let text = self.config.text.clone();
        let prefix = format!("layers.{layer_id}");
        let mix = (2 + text.hc_mult) * text.hc_mult;
        let width = text.hc_mult * text.hidden_size;
        let hyper = DeepseekV41HyperConnectionWeights {
            attn_fn: self.read_tensor(&format!("{prefix}.hc_attn_fn"), &[mix, width])?,
            attn_base: self.read_tensor(&format!("{prefix}.hc_attn_base"), &[mix])?,
            attn_scale: self.read_tensor(&format!("{prefix}.hc_attn_scale"), &[3])?,
            ffn_fn: self.read_tensor(&format!("{prefix}.hc_ffn_fn"), &[mix, width])?,
            ffn_base: self.read_tensor(&format!("{prefix}.hc_ffn_base"), &[mix])?,
            ffn_scale: self.read_tensor(&format!("{prefix}.hc_ffn_scale"), &[3])?,
        };
        let hidden = text.hidden_size;
        let inter = text.moe_intermediate_size;
        let storage = self.tensor_dtype(&format!("{prefix}.ffn.experts.0.w1.weight"))?;
        let moe = if residency != super::DeepseekV41ExpertResidency::Resident {
            super::backbone::MoeLayerWeights::Streamed(self.load_streamed_moe(
                layer_id,
                residency == super::DeepseekV41ExpertResidency::StreamedAllExperts,
            )?)
        } else if matches!(storage.as_str(), "BF16" | "F8_E4M3" | "F4" | "I8")
            || self.lora.as_ref().is_some_and(|adapter| {
                adapter
                    .targets()
                    .any(|(target, _, _)| target.starts_with(&format!("{prefix}.ffn.")))
            })
        {
            super::backbone::MoeLayerWeights::Mixed(self.load_mixed_moe(layer_id)?)
        } else {
            let mut moe = DeepseekV41MoeWeights {
                gate_weight: self.read_tensor(
                    &format!("{prefix}.ffn.gate.weight"),
                    &[text.n_routed_experts, hidden],
                )?,
                gate_bias: self
                    .read_tensor(&format!("{prefix}.ffn.gate.bias"), &[text.n_routed_experts])?,
                expert_w1: Vec::with_capacity(text.n_routed_experts),
                expert_w2: Vec::with_capacity(text.n_routed_experts),
                expert_w3: Vec::with_capacity(text.n_routed_experts),
                shared_w1: self.read_tensor(
                    &format!("{prefix}.ffn.shared_experts.w1.weight"),
                    &[inter, hidden],
                )?,
                shared_w2: self.read_tensor(
                    &format!("{prefix}.ffn.shared_experts.w2.weight"),
                    &[hidden, inter],
                )?,
                shared_w3: self.read_tensor(
                    &format!("{prefix}.ffn.shared_experts.w3.weight"),
                    &[inter, hidden],
                )?,
            };
            for expert in 0..text.n_routed_experts {
                let name = format!("{prefix}.ffn.experts.{expert}");
                moe.expert_w1
                    .push(self.read_tensor(&format!("{name}.w1.weight"), &[inter, hidden])?);
                moe.expert_w2
                    .push(self.read_tensor(&format!("{name}.w2.weight"), &[hidden, inter])?);
                moe.expert_w3
                    .push(self.read_tensor(&format!("{name}.w3.weight"), &[inter, hidden])?);
            }
            super::backbone::MoeLayerWeights::Fp32(moe)
        };
        Ok(super::backbone::BackboneLayerWeights {
            attention,
            hyper,
            moe,
            attn_norm: self.read_tensor(&format!("{prefix}.attn_norm.weight"), &[hidden])?,
            ffn_norm: self.read_tensor(&format!("{prefix}.ffn_norm.weight"), &[hidden])?,
        })
    }

    /// Engram tables must already carry the released post-dequantization BF16
    /// values. Packed table decoding and tokenizer/hash conversion are separate.
    pub fn load_engram(&mut self, layer_id: usize) -> Result<DeepseekV41EngramWeights> {
        let embedding = self.load_dense_engram_table(layer_id)?;
        let mut weights = self.load_engram_projection(layer_id)?;
        weights.embedding = embedding;
        Ok(weights)
    }

    /// Dense table payload only. Projection and Q/K norms are loaded separately
    /// so checkpoint-streamed tables never materialize beside their layer.
    pub fn load_dense_engram_table(&mut self, layer_id: usize) -> Result<Vec<f32>> {
        let layout = DeepseekV41EngramLayout::from_text(&self.config.text)?
            .context("V4.1 checkpoint has no Engram schedule")?;
        let index = layout
            .layer_hash_index(layer_id)
            .context("V4.1 layer does not own Engram")?;
        let embedding = self.read_tensor(
            &format!("layers.{layer_id}.engram.embed.weight"),
            &[layout.num_embeddings[index], layout.head_dim],
        )?;
        if embedding.iter().any(|value| value.to_bits() & 0xffff != 0) {
            bail!("V4.1 Engram embedding must contain BF16-exact post-dequantization values");
        }
        Ok(embedding)
    }

    /// Converted inference names, single-rank row layout. Only the table stays
    /// packed; this does not quantize or alter the projection/norm arithmetic.
    pub fn load_packed_engram(
        &mut self,
        layer_id: usize,
    ) -> Result<(DeepseekV41PackedWeights, DeepseekV41EngramWeights)> {
        Ok((
            self.load_engram_table(layer_id)?,
            self.load_engram_projection(layer_id)?,
        ))
    }

    /// Packed table payload only. Projection and Q/K norms are loaded separately
    /// so checkpoint-streamed tables never materialize beside their layer.
    pub fn load_engram_table(&mut self, layer_id: usize) -> Result<DeepseekV41PackedWeights> {
        let layout = DeepseekV41EngramLayout::from_text(&self.config.text)?
            .context("V4.1 checkpoint has no Engram schedule")?;
        let index = layout
            .layer_hash_index(layer_id)
            .context("V4.1 layer does not own Engram")?;
        let prefix = format!("layers.{layer_id}.engram.embed");
        self.read_packed_weight(
            &format!("{prefix}.weight"),
            &format!("{prefix}.scale"),
            layout.num_embeddings[index],
            layout.head_dim,
            DeepseekV41WeightFormat::EngramFp8Row32,
        )
    }

    pub fn load_engram_projection(&mut self, layer_id: usize) -> Result<DeepseekV41EngramWeights> {
        let text = self.config.text.clone();
        let layout = DeepseekV41EngramLayout::from_text(&text)?
            .context("V4.1 checkpoint has no Engram schedule")?;
        if layout.layer_hash_index(layer_id).is_none() {
            bail!("V4.1 layer does not own Engram");
        }
        let prefix = format!("layers.{layer_id}.engram");
        let wkv = self.read_tensor(
            &format!("{prefix}.wkv.weight"),
            &[
                text.hidden_size * (text.hc_mult + 1),
                layout.n_hash_cols() * layout.head_dim,
            ],
        )?;
        let mut weights = self.load_engram_norms(layer_id)?;
        weights.wkv = wkv;
        Ok(weights)
    }

    /// Typed projection plus resident packed table for the hybrid model path.
    pub fn load_mixed_engram(
        &mut self,
        layer_id: usize,
    ) -> Result<(
        DeepseekV41PackedWeights,
        DeepseekV41LinearWeights,
        DeepseekV41EngramWeights,
    )> {
        let table = self.load_engram_table(layer_id)?;
        let (projection, weights) = self.load_mixed_engram_projection(layer_id)?;
        Ok((table, projection, weights))
    }

    /// Typed packed projection and Q/K norms without the table.
    pub fn load_mixed_engram_projection(
        &mut self,
        layer_id: usize,
    ) -> Result<(DeepseekV41LinearWeights, DeepseekV41EngramWeights)> {
        let text = self.config.text.clone();
        let layout = DeepseekV41EngramLayout::from_text(&text)?
            .context("V4.1 checkpoint has no Engram schedule")?;
        layout
            .layer_hash_index(layer_id)
            .context("V4.1 layer does not own Engram")?;
        let projection = self.read_linear_weight(
            &format!("layers.{layer_id}.engram.wkv.weight"),
            text.hidden_size * (text.hc_mult + 1),
            layout.n_hash_cols() * layout.head_dim,
            DeepseekV41WeightFormat::Fp8Block32,
        )?;
        Ok((projection, self.load_engram_norms(layer_id)?))
    }

    fn load_engram_norms(&mut self, layer_id: usize) -> Result<DeepseekV41EngramWeights> {
        let text = &self.config.text;
        let shape = [text.hc_mult, text.hidden_size];
        let prefix = format!("layers.{layer_id}.engram");
        Ok(DeepseekV41EngramWeights {
            embedding: vec![],
            wkv: vec![],
            q_weight: self.read_tensor(&format!("{prefix}.q_weight"), &shape)?,
            k_weight: self.read_tensor(&format!("{prefix}.k_weight"), &shape)?,
        })
    }

    pub fn load_compressor(&mut self, layer_id: usize) -> Result<DeepseekV41CompressorWeights> {
        let plan = self.config.plan()?;
        let layer = plan
            .backbone
            .get(layer_id)
            .context("V4.1 compressor layer is out of range")?;
        if layer.kv_source_layer != Some(layer_id) {
            bail!("V4.1 layer {layer_id} does not own a compressor");
        }
        let hidden = self.config.text.hidden_size;
        let head = self.config.text.head_dim;
        let prefix = format!("layers.{layer_id}.attn.compressor");
        Ok(DeepseekV41CompressorWeights {
            hidden_size: hidden,
            head_dim: head,
            compress_ratio: layer.compress_ratio,
            eps: self.config.text.rms_norm_eps as f32,
            kv_weight: self.read_tensor(&format!("{prefix}.wkv.weight"), &[head, hidden])?,
            gate_weight: if layer.compress_ratio > 1 {
                Some(self.read_tensor(&format!("{prefix}.wgate.weight"), &[head, hidden])?)
            } else {
                None
            },
            norm_weight: self.read_tensor(&format!("{prefix}.norm.weight"), &[head])?,
        })
    }
}

/// One layer's checkpoint-backed projection source. Components created with a
/// streamed projection policy hold this instead of uploaded layer weights and
/// read the layer's payload when the orchestrator enters that layer.
pub(super) struct DeepseekV41StreamedLayer {
    checkpoint: std::sync::Mutex<DeepseekV41Checkpoint>,
    layer: usize,
}

impl DeepseekV41StreamedLayer {
    pub(super) fn new(checkpoint: DeepseekV41Checkpoint, layer: usize) -> Self {
        Self {
            checkpoint: std::sync::Mutex::new(checkpoint),
            layer,
        }
    }

    pub(super) fn layer(&self) -> usize {
        self.layer
    }

    pub(super) fn lock(&self) -> Result<std::sync::MutexGuard<'_, DeepseekV41Checkpoint>> {
        self.checkpoint
            .lock()
            .map_err(|_| anyhow::anyhow!("V4.1 streamed layer checkpoint lock poisoned"))
    }
}
