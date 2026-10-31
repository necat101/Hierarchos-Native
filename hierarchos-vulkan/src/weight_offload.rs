//! Architecture-agnostic tiered weight residency.
//!
//! Every native graph loads released weights the same way: a SafeTensors package
//! arrives on disk, some tensors are decoded on the host, and the values that a
//! forward or backward pass actually touches become device buffers. Those three
//! places are the tiers this module names and makes detectable:
//!
//! * **device** - a tensor or tile uploaded into a `GpuBuffer`.
//! * **host** - a bounded cache of decoded payloads, sized by an explicit byte
//!   budget rather than by "however much the model happens to be".
//! * **package** - the SafeTensors file, read through the OS page cache.
//!
//! ## Why this exists rather than each loader doing its own I/O
//!
//! The pre-existing host readers called `fs::read(path)` and then deserialised
//! the whole shard for every single tensor they wanted. A twenty-shard package
//! therefore read and parsed the entire package once per requested tensor, and
//! the decoded `Vec<f32>` of the largest tensors was materialised in full even
//! when only a row range was needed. Both costs scale with the package, not with
//! the request, which is exactly what stops a large released checkpoint from
//! running on a small device.
//!
//! This module reads *only the requested byte range* of *only the requested
//! tensor*, keeps the parsed headers in a process-wide cache, and can serve
//! repeated row reads from a byte-capped host tile cache. Values, dtypes and
//! arithmetic are untouched: the same bytes decode to the same `f32`s.
//!
//! ## The residency decision
//!
//! [`plan_weight_residency`] answers the question every loader has to answer
//! before it allocates: does this parameter set fit beside what is already
//! resident? `Resident` means the caller may hold the whole tensor (or every
//! tile of it) on the device at once; `Tiered` means it must stage pieces
//! through host memory. An unknown budget is treated as not fitting, so an
//! unreadable driver budget degrades to the bounded path rather than to an
//! out-of-memory failure.
//!
//! The ledger records what actually happened, so a claim about tiers can be
//! checked instead of asserted.

use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::VulkanDevice;

/// SafeTensors header length field width and the largest header this module
/// accepts. The format stores the header length as a little-endian `u64`
/// immediately before the JSON header.
const HEADER_LENGTH_BYTES: u64 = 8;
/// A released package's header is a few megabytes of JSON at most. Anything past
/// this is a corrupt or hostile file rather than a package.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
/// Default target size of one cached host tile. Matches the released-width
/// vocabulary tiling that the dedicated V4.1 graph already uses, so both paths
/// stage comparable amounts through host memory.
pub const DEFAULT_TILE_BYTES: u64 = 8 * 1024 * 1024;
/// Default host tile-cache capacity. Four tiles is enough to keep an embedding
/// gather or a head projection from re-reading the same rows every step while
/// staying a rounding error beside a released checkpoint.
pub const DEFAULT_CACHE_BYTES: u64 = 4 * DEFAULT_TILE_BYTES;
/// Device headroom left free before a parameter set is accepted as resident.
pub const DEFAULT_DEVICE_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// Storage format of one package tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDtype {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    U32,
    U8,
}

impl TensorDtype {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "F64" => Self::F64,
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "U8" => Self::U8,
            other => bail!("unsupported SafeTensors storage {other}"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::U8 => "U8",
        }
    }

    pub fn item_bytes(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::U8 => 1,
        }
    }

    /// Convert from the `safetensors` crate's enum, so a caller holding a
    /// resolved tensor source can hand it to this module's bounded readers.
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        Ok(match dtype {
            safetensors::Dtype::F64 => Self::F64,
            safetensors::Dtype::F32 => Self::F32,
            safetensors::Dtype::F16 => Self::F16,
            safetensors::Dtype::BF16 => Self::Bf16,
            safetensors::Dtype::I64 => Self::I64,
            safetensors::Dtype::I32 => Self::I32,
            safetensors::Dtype::U32 => Self::U32,
            safetensors::Dtype::U8 => Self::U8,
            other => bail!("unsupported SafeTensors storage {other:?}"),
        })
    }

    /// Convert to the `safetensors` crate's equivalent, so a caller that still
    /// matches on that enum keeps one code path for both readers.
    pub fn to_safetensors(self) -> safetensors::Dtype {
        match self {
            Self::F64 => safetensors::Dtype::F64,
            Self::F32 => safetensors::Dtype::F32,
            Self::F16 => safetensors::Dtype::F16,
            Self::Bf16 => safetensors::Dtype::BF16,
            Self::I64 => safetensors::Dtype::I64,
            Self::I32 => safetensors::Dtype::I32,
            Self::U32 => safetensors::Dtype::U32,
            Self::U8 => safetensors::Dtype::U8,
        }
    }
}

#[derive(serde::Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Where one tensor lives, resolved from the shard header alone. No payload is
/// read to produce this.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub shard: PathBuf,
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

impl serde::Serialize for TensorEntry {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TensorEntry", 5)?;
        state.serialize_field("shard", &self.shard.display().to_string())?;
        state.serialize_field("dtype", &self.dtype)?;
        state.serialize_field("shape", &self.shape)?;
        state.serialize_field("data_offsets", &self.data_offsets)?;
        state.serialize_field("byte_len", &self.byte_len().unwrap_or(0))?;
        state.end()
    }
}

impl TensorEntry {
    pub fn element_count(&self) -> Result<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .context("package tensor element count overflow")
    }

    pub fn byte_len(&self) -> Result<u64> {
        let bytes = (self.element_count()?)
            .checked_mul(self.dtype.item_bytes())
            .context("package tensor byte count overflow")?;
        Ok(bytes as u64)
    }

    /// Row width for a two-dimensional tensor, in bytes.
    pub fn row_bytes(&self) -> Result<usize> {
        match self.shape.as_slice() {
            [_, cols] => cols
                .checked_mul(self.dtype.item_bytes())
                .context("package tensor row byte count overflow"),
            _ => bail!(
                "package row selection requires a 2-D tensor, got {:?}",
                self.shape
            ),
        }
    }
}

/// One shard's parsed header. Shared between every reader in the process, since
/// a package's headers are immutable for as long as the file is.
struct ShardHeader {
    path: PathBuf,
    data_start: u64,
    file_size: u64,
    tensors: BTreeMap<String, (TensorDtype, Vec<usize>, [u64; 2])>,
}

fn header_cache() -> &'static Mutex<HashMap<(PathBuf, u64, i64), Arc<ShardHeader>>> {
    static CACHE: OnceLock<Mutex<HashMap<(PathBuf, u64, i64), Arc<ShardHeader>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Identity of a shard file on disk. Regenerated packages therefore get a fresh
/// header instead of a stale one.
fn shard_identity(path: &Path) -> Result<(u64, i64)> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("reading {} metadata", path.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|elapsed| elapsed.as_nanos() as i64)
        .unwrap_or(0);
    Ok((metadata.len(), modified))
}

/// Read and parse one SafeTensors header, caching the result process-wide.
fn shard_header(path: &Path) -> Result<Arc<ShardHeader>> {
    let identity = shard_identity(path)?;
    if let Ok(cache) = header_cache().lock() {
        if let Some(header) = cache.get(&(path.to_path_buf(), identity.0, identity.1)) {
            return Ok(header.clone());
        }
    }
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut length = [0u8; HEADER_LENGTH_BYTES as usize];
    file.read_exact(&mut length)
        .with_context(|| format!("reading SafeTensors header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(length);
    let data_start = header_len
        .checked_add(HEADER_LENGTH_BYTES)
        .context("SafeTensors header overflow")?;
    if header_len > MAX_HEADER_BYTES || data_start > file_size {
        bail!(
            "{} has an invalid or oversized SafeTensors header",
            path.display()
        );
    }
    let mut raw = vec![0u8; header_len as usize];
    file.read_exact(&mut raw)?;
    let mut entries: BTreeMap<String, Value> = serde_json::from_slice(&raw)
        .with_context(|| format!("decoding SafeTensors header of {}", path.display()))?;
    entries.remove("__metadata__");
    let mut tensors = BTreeMap::new();
    for (name, value) in entries {
        let info: RawTensorInfo = serde_json::from_value(value)
            .with_context(|| format!("decoding SafeTensors entry {name} of {}", path.display()))?;
        tensors.insert(
            name,
            (
                TensorDtype::parse(&info.dtype)?,
                info.shape,
                info.data_offsets,
            ),
        );
    }
    let header = Arc::new(ShardHeader {
        path: path.to_path_buf(),
        data_start,
        file_size,
        tensors,
    });
    if let Ok(mut cache) = header_cache().lock() {
        cache.insert((path.to_path_buf(), identity.0, identity.1), header.clone());
    }
    Ok(header)
}

/// Byte accounting for one reader. Every number here is a count of bytes or
/// events that actually happened, so a tier claim can be checked.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
pub struct OffloadLedger {
    /// Shard headers parsed (cache misses on the header cache).
    pub headers_parsed: u64,
    /// Bytes of tensor payload pulled from the package file.
    pub payload_bytes_read: u64,
    /// Payload bytes returned from the host tile cache without touching disk.
    pub payload_bytes_from_cache: u64,
    /// Bytes of tuple bytes decoded into `f32`, `f64`, ... host vectors.
    pub decoded_bytes: u64,
    /// Largest single returned f32 vector, including a caller's whole-tensor
    /// materialization. This measures decoded request size rather than cache size.
    pub max_f32_request_bytes: u64,
    /// Largest individual raw payload read issued to the filesystem.
    pub max_payload_read_bytes: u64,
    /// Individual tensor payload reads issued.
    pub payload_reads: u64,
    /// Read requests that were satisfied entirely from the host cache.
    pub cache_hits: u64,
    /// Read requests that needed at least one package read.
    pub cache_misses: u64,
    /// Tiles dropped to respect the host cache byte budget.
    pub tiles_evicted: u64,
    /// Peak host tile-cache occupancy in bytes.
    pub cache_peak_bytes: u64,
    /// Whole-tensor reads (as opposed to bounded row ranges).
    pub whole_tensor_reads: u64,
    /// Bounded row-range reads.
    pub row_range_reads: u64,
    /// Row-gather reads: an arbitrary, possibly repeated set of row ids.
    pub row_gathers: u64,
    /// Distinct rows a gather actually pulled, counted once per cache miss.
    pub rows_read: u64,
}

impl OffloadLedger {
    pub fn host_bytes(&self) -> u64 {
        self.payload_bytes_read + self.payload_bytes_from_cache
    }

    /// Fraction of served payload bytes that never reached the package file.
    pub fn cache_hit_ratio(&self) -> f64 {
        let served = self.host_bytes();
        if served == 0 {
            0.0
        } else {
            self.payload_bytes_from_cache as f64 / served as f64
        }
    }
}

/// Bounded cache of decoded package payload tiles, keyed by shard, tensor and
/// tile index. Occupancy never exceeds `capacity_bytes`; the tile least recently
/// inserted or touched is dropped to make room.
struct HostTileCache {
    capacity_bytes: u64,
    resident_bytes: u64,
    clock: u64,
    tiles: HashMap<TileKey, CachedTile>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TileKey {
    shard: PathBuf,
    tensor: String,
    // Physical byte ranges allow row and flat views to share identical tiles,
    // without confusing a row index with an element index for the same tensor.
    byte_offset: usize,
    byte_len: usize,
}

struct CachedTile {
    bytes: Arc<Vec<u8>>,
    last_used: u64,
}

impl HostTileCache {
    fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            clock: 0,
            tiles: HashMap::new(),
        }
    }

    fn get(&mut self, key: &TileKey) -> Option<Arc<Vec<u8>>> {
        self.clock = self.clock.wrapping_add(1);
        let clock = self.clock;
        let tile = self.tiles.get_mut(key)?;
        tile.last_used = clock;
        Some(tile.bytes.clone())
    }

    fn insert(&mut self, key: TileKey, bytes: Arc<Vec<u8>>, ledger: &mut OffloadLedger) {
        if self.capacity_bytes == 0 {
            return;
        }
        let incoming = bytes.len() as u64;
        if incoming > self.capacity_bytes {
            // A single tile larger than the whole budget is never retained;
            // the reader still returns it to its caller.
            return;
        }
        if self.tiles.remove(&key).is_some() {
            self.resident_bytes = self.resident_bytes.saturating_sub(incoming);
        }
        while self.resident_bytes + incoming > self.capacity_bytes {
            let Some(victim) = self
                .tiles
                .iter()
                .min_by_key(|(_, tile)| tile.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(dropped) = self.tiles.remove(&victim) {
                self.resident_bytes = self
                    .resident_bytes
                    .saturating_sub(dropped.bytes.len() as u64);
                ledger.tiles_evicted += 1;
            }
        }
        self.resident_bytes += incoming;
        ledger.cache_peak_bytes = ledger.cache_peak_bytes.max(self.resident_bytes);
        self.clock = self.clock.wrapping_add(1);
        self.tiles.insert(
            key,
            CachedTile {
                bytes,
                last_used: self.clock,
            },
        );
    }
}

/// A readable view of one native model package.
///
/// `open_directory` follows the released sharded layout (`model.safetensors`
/// plus an optional `model.safetensors.index.json` weight map); `open_file`
/// reads a single-file package. Both resolve tensor locations from headers only.
pub struct ModelPackage {
    root: PathBuf,
    /// The exact file a single-file package was opened from. Kept separately
    /// from `root` because the caller's filename is not necessarily
    /// `model.safetensors`.
    single_file: Option<PathBuf>,
    weight_map: Option<Arc<BTreeMap<String, String>>>,
    shards: BTreeMap<PathBuf, Arc<ShardHeader>>,
    cache: HostTileCache,
    tile_bytes: u64,
    ledger: OffloadLedger,
}

impl ModelPackage {
    /// Open a single-file SafeTensors package. `path` may be the file itself or
    /// a directory containing `model.safetensors`.
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_path_buf()
        };
        if !file.is_file() {
            bail!("no SafeTensors package at {}", file.display());
        }
        let root = file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self {
            root,
            single_file: Some(file),
            weight_map: None,
            shards: BTreeMap::new(),
            cache: HostTileCache::new(DEFAULT_CACHE_BYTES),
            tile_bytes: DEFAULT_TILE_BYTES,
            ledger: OffloadLedger::default(),
        })
    }

    /// Open a released sharded package directory.
    pub fn open_directory(directory: impl AsRef<Path>) -> Result<Self> {
        let root = directory
            .as_ref()
            .canonicalize()
            .with_context(|| format!("opening package {}", directory.as_ref().display()))?;
        let index = root.join("model.safetensors.index.json");
        let weight_map = if index.is_file() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            let index: Index = serde_json::from_slice(
                &std::fs::read(&index).with_context(|| format!("reading {}", index.display()))?,
            )?;
            Some(Arc::new(index.weight_map))
        } else {
            if !root.join("model.safetensors").is_file() {
                bail!(
                    "package {} has neither model.safetensors.index.json nor model.safetensors",
                    root.display()
                );
            }
            None
        };
        Ok(Self {
            root,
            single_file: None,
            weight_map,
            shards: BTreeMap::new(),
            cache: HostTileCache::new(DEFAULT_CACHE_BYTES),
            tile_bytes: DEFAULT_TILE_BYTES,
            ledger: OffloadLedger::default(),
        })
    }

    /// Open the package a single shard path belongs to, preferring the sharded
    /// layout when the sibling index exists.
    pub fn open_containing(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::open_directory(path);
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        if parent.join("model.safetensors.index.json").is_file() {
            return Self::open_directory(parent);
        }
        Self::open_file(path)
    }

    /// Set the host tile cache byte budget. Zero disables retention.
    pub fn with_cache_bytes(mut self, bytes: u64) -> Self {
        self.cache = HostTileCache::new(bytes);
        self
    }

    /// Set the target host tile size used to align bounded row reads.
    pub fn with_tile_bytes(mut self, bytes: u64) -> Self {
        self.tile_bytes = bytes.max(1);
        self
    }

    pub fn ledger(&self) -> OffloadLedger {
        self.ledger
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve the shard that owns `name`, parsing its header on first use.
    fn shard_for(&mut self, name: &str) -> Result<Arc<ShardHeader>> {
        // A single-file package addresses exactly the file the caller named.
        let path = if let Some(file) = &self.single_file {
            file.clone()
        } else {
            let filename = match &self.weight_map {
                Some(index) => index
                    .get(name)
                    .with_context(|| format!("package index has no tensor {name}"))?
                    .clone(),
                None => "model.safetensors".to_owned(),
            };
            let relative = Path::new(&filename);
            if relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            {
                bail!("package shard must be a relative path within the package: {filename}");
            }
            if self.weight_map.is_some() {
                let joined = self
                    .root
                    .join(relative)
                    .canonicalize()
                    .with_context(|| format!("opening package shard {filename}"))?;
                if !joined.starts_with(&self.root) {
                    bail!("package shard resolves outside the package: {filename}");
                }
                joined
            } else {
                self.root.join(relative)
            }
        };
        if let Some(header) = self.shards.get(&path) {
            return Ok(header.clone());
        }
        let header = shard_header(&path)?;
        self.ledger.headers_parsed += 1;
        self.shards.insert(path, header.clone());
        Ok(header)
    }

    /// Whether the package declares `name`.
    ///
    /// A missing tensor is `false`; an unreadable shard is still an error, so a
    /// caller probing optional tensors cannot mistake a broken file for an
    /// absent one.
    pub fn contains(&mut self, name: &str) -> Result<bool> {
        if let Some(index) = &self.weight_map {
            return Ok(index.contains_key(name));
        }
        let header = self.shard_for(name)?;
        Ok(header.tensors.contains_key(name))
    }

    /// Resolve a tensor's location, rejecting a header whose declared shape and
    /// payload offsets disagree, or that places the payload past the shard's
    /// data section. Both are integrity failures, not layout choices.
    pub fn entry(&mut self, name: &str) -> Result<TensorEntry> {
        let header = self.shard_for(name)?;
        let (dtype, shape, data_offsets) = header
            .tensors
            .get(name)
            .with_context(|| format!("package {} is missing {name}", header.path.display()))?;
        let entry = TensorEntry {
            shard: header.path.clone(),
            dtype: *dtype,
            shape: shape.clone(),
            data_offsets: *data_offsets,
        };
        let bytes = entry.byte_len()?;
        let span = data_offsets[1]
            .checked_sub(data_offsets[0])
            .context("package tensor payload offsets are inverted")?;
        if span != bytes {
            bail!(
                "package tensor {name} declares {bytes} payload bytes but its offsets span {span}"
            );
        }
        if data_offsets[1] > header.file_size.saturating_sub(header.data_start) {
            bail!("package tensor {name} is placed past the end of its shard data section");
        }
        Ok(entry)
    }

    pub fn dtype(&mut self, name: &str) -> Result<TensorDtype> {
        Ok(self.entry(name)?.dtype)
    }

    pub fn shape(&mut self, name: &str) -> Result<Vec<usize>> {
        Ok(self.entry(name)?.shape)
    }

    pub fn tensor_bytes(&mut self, name: &str) -> Result<u64> {
        self.entry(name)?.byte_len()
    }

    /// Every tensor name the package declares, header-only: no payload is read.
    ///
    /// A caller that has to *find* a tensor it was not told the name of - the
    /// streamed output head is one - needs the declared set rather than a guess.
    /// The order is the weight map's (or the single shard header's), so callers
    /// that want a deterministic pick must sort it themselves.
    pub fn tensor_names(&mut self) -> Result<Vec<String>> {
        let mut names: Vec<String> = match &self.weight_map {
            Some(index) => index.keys().cloned().collect(),
            None => {
                let header = self.shard_for("model.safetensors")?;
                header.tensors.keys().cloned().collect()
            }
        };
        names.sort();
        Ok(names)
    }

    /// Total stored payload bytes declared by every shard reachable from this
    /// package's weight map. Header-only: no payload is read.
    pub fn stored_bytes(&mut self) -> Result<u64> {
        let names: Vec<String> = match &self.weight_map {
            Some(index) => index.keys().cloned().collect(),
            None => {
                let header = self.shard_for("model.safetensors")?;
                header.tensors.keys().cloned().collect()
            }
        };
        let mut total = 0u64;
        for name in names {
            total = total.saturating_add(self.tensor_bytes(&name)?);
        }
        Ok(total)
    }

    /// Read one tensor's complete payload from the package.
    pub fn read_payload(&mut self, name: &str) -> Result<(TensorDtype, Vec<usize>, Vec<u8>)> {
        let entry = self.entry(name)?;
        let bytes = self.read_payload_entry(&entry)?;
        self.ledger.whole_tensor_reads += 1;
        Ok((entry.dtype, entry.shape, bytes))
    }

    /// Read an arbitrary set of rows of a 2-D tensor.
    ///
    /// Rows are grouped by host tile, so a tile that contains several wanted
    /// rows is pulled from the package once for the whole request and, with
    /// retention enabled, once for the cache's lifetime. Output rows follow
    /// `row_ids` order and duplicates are served from the one read.
    ///
    /// A gather is the shape a vocabulary-sized table is actually consumed in:
    /// an embedding lookup wants the rows for this step's tokens, not the
    /// table. Returning only the requested rows is what keeps the caller's
    /// buffer proportional to the request rather than to the package tensor.
    pub fn read_row_gather(&mut self, name: &str, row_ids: &[usize]) -> Result<Vec<u8>> {
        let entry = self.entry(name)?;
        let total_rows = match entry.shape.as_slice() {
            [rows, _] => *rows,
            _ => bail!(
                "row gather requires a 2-D tensor, got {:?} for {name}",
                entry.shape
            ),
        };
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }
        let row_bytes = entry.row_bytes()?;
        let tile_rows = (self.tile_bytes / row_bytes.max(1) as u64).max(1) as usize;
        let mut wanted: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut destinations: HashMap<usize, Vec<usize>> = HashMap::new();
        for (output_row, &row) in row_ids.iter().enumerate() {
            if row >= total_rows {
                bail!("row gather index {row} is outside {name} with {total_rows} rows");
            }
            wanted
                .entry(row / tile_rows * tile_rows)
                .or_default()
                .push(row);
            destinations.entry(row).or_default().push(output_row);
        }
        let mut output = vec![0u8; row_ids.len() * row_bytes];
        for (tile_first, rows) in wanted {
            let tile_last = (tile_first + tile_rows).min(total_rows);
            let key = TileKey {
                shard: entry.shard.clone(),
                tensor: name.to_owned(),
                byte_offset: tile_first * row_bytes,
                byte_len: (tile_last - tile_first) * row_bytes,
            };
            let tile = match self.cache.get(&key) {
                Some(tile) => {
                    self.ledger.cache_hits += 1;
                    self.ledger.payload_bytes_from_cache = self
                        .ledger
                        .payload_bytes_from_cache
                        .saturating_add(tile.len() as u64);
                    tile
                }
                None => {
                    self.ledger.cache_misses += 1;
                    let tile = Arc::new(self.read_shard_rows(
                        &entry,
                        tile_first,
                        tile_last - tile_first,
                    )?);
                    self.ledger.rows_read = self
                        .ledger
                        .rows_read
                        .saturating_add((tile_last - tile_first) as u64);
                    self.cache.insert(key, tile.clone(), &mut self.ledger);
                    tile
                }
            };
            for row in rows {
                let from = (row - tile_first) * row_bytes;
                let source = &tile[from..from + row_bytes];
                for &output_row in &destinations[&row] {
                    let to = output_row * row_bytes;
                    output[to..to + row_bytes].copy_from_slice(source);
                }
            }
        }
        self.ledger.row_gathers += 1;
        Ok(output)
    }

    /// Read a bounded row range of a 2-D tensor, aligned to host tiles so that
    /// repeated reads of nearby rows reuse the cache.
    pub fn read_rows(&mut self, name: &str, first_row: usize, rows: usize) -> Result<Vec<u8>> {
        let entry = self.entry(name)?;
        let [total_rows, _] = match entry.shape.as_slice() {
            [rows, cols] => [*rows, *cols],
            _ => bail!("row selection requires a 2-D tensor, got {:?}", entry.shape),
        };
        if rows == 0 {
            return Ok(Vec::new());
        }
        let end_row = first_row
            .checked_add(rows)
            .context("package row range overflow")?;
        if end_row > total_rows {
            bail!("row range {first_row}..{end_row} is outside {name} with {total_rows} rows");
        }
        let row_bytes = entry.row_bytes()?;
        let tile_rows = (self.tile_bytes / row_bytes.max(1) as u64).max(1) as usize;
        let mut out = Vec::with_capacity(rows * row_bytes);
        let mut row = first_row;
        while row < end_row {
            let tile_first = row / tile_rows * tile_rows;
            let tile_last = (tile_first + tile_rows).min(total_rows);
            let key = TileKey {
                shard: entry.shard.clone(),
                tensor: name.to_owned(),
                byte_offset: tile_first * row_bytes,
                byte_len: (tile_last - tile_first) * row_bytes,
            };
            let tile = match self.cache.get(&key) {
                Some(tile) => {
                    self.ledger.cache_hits += 1;
                    self.ledger.payload_bytes_from_cache = self
                        .ledger
                        .payload_bytes_from_cache
                        .saturating_add(tile.len() as u64);
                    tile
                }
                None => {
                    self.ledger.cache_misses += 1;
                    let tile = Arc::new(self.read_shard_rows(
                        &entry,
                        tile_first,
                        tile_last - tile_first,
                    )?);
                    self.cache.insert(key, tile.clone(), &mut self.ledger);
                    tile
                }
            };
            let take_end = end_row.min(tile_last);
            let from = (row - tile_first) * row_bytes;
            let to = from + (take_end - row) * row_bytes;
            out.extend_from_slice(&tile[from..to]);
            row = take_end;
        }
        self.ledger.row_range_reads += 1;
        Ok(out)
    }

    /// Read a bounded row range of a 2-D tensor and decode it to `f32`.
    ///
    /// This is the read a vocabulary-sized projection tile needs: `first_row`
    /// selects the output rows the tile covers, the host tile cache keeps the
    /// read proportional to the tile rather than to the table, and the decode is
    /// the same one every other reader uses.
    pub fn read_f32_rows(&mut self, name: &str, first_row: usize, rows: usize) -> Result<Vec<f32>> {
        let entry = self.entry(name)?;
        let raw = self.read_rows(name, first_row, rows)?;
        self.ledger.decoded_bytes = self.ledger.decoded_bytes.saturating_add(raw.len() as u64);
        self.ledger.max_f32_request_bytes = self.ledger.max_f32_request_bytes.max((raw.len() / entry.dtype.item_bytes() * 4) as u64);
        Self::decode_f32(entry.dtype, &raw)
    }

    /// Bounded contiguous view into a packed floating tensor's logical element
    /// order (for example one expert in a rank-3 gate/up parameter). No shape
    /// reinterpretation or value conversion beyond the ordinary dtype decode.
    pub fn read_f32_elements(
        &mut self,
        name: &str,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>> {
        let entry = self.entry(name)?;
        if first
            .checked_add(count)
            .is_none_or(|end| end > entry.element_count().unwrap_or(0))
        {
            bail!("flat range outside package tensor {name}");
        }
        self.ledger.max_f32_request_bytes = self
            .ledger
            .max_f32_request_bytes
            .max(count.saturating_mul(4) as u64);
        let item = entry.dtype.item_bytes();
        let cache_bytes = self.cache.capacity_bytes;
        let chunk = (if cache_bytes == 0 {
            self.tile_bytes
        } else {
            self.tile_bytes.min(cache_bytes)
        } as usize
            / item)
            .max(1);
        let mut values = Vec::with_capacity(count);
        let mut done = 0;
        while done < count {
            let at = first + done;
            // Cache-disabled callers retain exact-range I/O; enabled readers
            // share bounded physical tiles across packed expert and split views.
            let tile_first = if cache_bytes == 0 {
                at
            } else {
                at / chunk * chunk
            };
            let tile_len = chunk.min(entry.element_count()? - tile_first);
            let n = (count - done).min(tile_len - (at - tile_first));
            let byte_len = if cache_bytes == 0 {
                n * item
            } else {
                tile_len * item
            };
            let key = TileKey {
                shard: entry.shard.clone(),
                tensor: name.to_owned(),
                byte_offset: tile_first * item,
                byte_len,
            };
            let raw = match self.cache.get(&key) {
                Some(tile) => {
                    self.ledger.cache_hits += 1;
                    self.ledger.payload_bytes_from_cache = self
                        .ledger
                        .payload_bytes_from_cache
                        .saturating_add(tile.len() as u64);
                    tile
                }
                None => {
                    self.ledger.cache_misses += 1;
                    let tile = Arc::new(self.read_shard_range(
                        &entry,
                        key.byte_offset as u64,
                        byte_len,
                    )?);
                    self.cache.insert(key, tile.clone(), &mut self.ledger);
                    tile
                }
            };
            let from = (at - tile_first) * item;
            self.ledger.decoded_bytes = self.ledger.decoded_bytes.saturating_add((n * item) as u64);
            values.extend(Self::decode_f32(entry.dtype, &raw[from..from + n * item])?);
            done += n;
        }
        self.ledger.row_range_reads += 1;
        Ok(values)
    }
    /// Materialize a whole 2-D tensor by walking it in bounded row ranges.
    ///
    /// Used where a complete table is genuinely required - parameter export, for
    /// example - so even that path reads through the byte-capped tiles instead of
    /// one unbounded allocation.
    pub fn read_all_f32_rows(&mut self, name: &str) -> Result<Vec<f32>> {
        let shape = self.shape(name)?;
        let [rows, _] = shape.as_slice() else {
            bail!("row materialization requires a 2-D tensor, got {shape:?}");
        };
        self.read_f32_row_prefix(name, *rows)
    }

    /// Materialize the logical vocabulary prefix through bounded row reads.
    /// Reserved physical rows remain outside this parameter's export and gather
    /// geometry; callers that preserve a wrapper can restore them from its source.
    pub fn read_f32_row_prefix(&mut self, name: &str, rows: usize) -> Result<Vec<f32>> {
        let entry = self.entry(name)?;
        let total_rows = match entry.shape.as_slice() {
            [rows, cols] => {
                if *cols == 0 {
                    bail!("tensor {name} has zero columns");
                }
                *rows
            }
            _ => bail!(
                "row materialization requires a 2-D tensor, got {:?}",
                entry.shape
            ),
        };
        if rows > total_rows {
            bail!("tensor {name} has {total_rows} rows; cannot materialize a {rows}-row prefix");
        }
        self.ledger.max_f32_request_bytes = self.ledger.max_f32_request_bytes.max(rows.saturating_mul(entry.shape[1]).saturating_mul(4) as u64);
        let row_bytes = entry.row_bytes()?;
        let tile_rows = (self.tile_bytes / row_bytes.max(1) as u64).max(1) as usize;
        let mut values = Vec::new();
        let mut row = 0usize;
        while row < rows {
            let count = tile_rows.min(rows - row);
            let tile = self.read_rows(name, row, count)?;
            self.ledger.decoded_bytes = self.ledger.decoded_bytes.saturating_add(tile.len() as u64);
            values.extend(Self::decode_f32(entry.dtype, &tile)?);
            row += count;
        }
        Ok(values)
    }

    fn read_shard_rows(
        &mut self,
        entry: &TensorEntry,
        first_row: usize,
        rows: usize,
    ) -> Result<Vec<u8>> {
        let row_bytes = entry.row_bytes()?;
        let length = rows
            .checked_mul(row_bytes)
            .context("package row range byte count overflow")?;
        let offset = (first_row as u64)
            .checked_mul(row_bytes as u64)
            .context("package row offset overflow")?;
        self.read_shard_range(entry, offset, length)
    }

    fn read_shard_range(
        &mut self,
        entry: &TensorEntry,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        let header = shard_header(&entry.shard)?;
        let [begin, _] = entry.data_offsets;
        let absolute = header
            .data_start
            .checked_add(begin)
            .and_then(|start| start.checked_add(offset))
            .context("package read offset overflow")?;
        if absolute
            .checked_add(length as u64)
            .is_none_or(|end| end > header.file_size)
        {
            bail!(
                "package read of {length} bytes at {absolute} runs past {}",
                header.path.display()
            );
        }
        let mut file = File::open(&header.path)
            .with_context(|| format!("opening {}", header.path.display()))?;
        file.seek(SeekFrom::Start(absolute))?;
        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer)?;
        self.ledger.payload_bytes_read =
            self.ledger.payload_bytes_read.saturating_add(length as u64);
        self.ledger.payload_reads += 1;
        self.ledger.max_payload_read_bytes=self.ledger.max_payload_read_bytes.max(length as u64);
        Ok(buffer)
    }

    /// Materialise a one-tensor SafeTensors image containing exactly `name`'s
    /// payload.
    ///
    /// This lets a reader keep its existing decode body verbatim while paying
    /// only for the tensor it asked for: the returned buffer is a valid
    /// SafeTensors file whose single entry carries the original dtype and shape,
    /// so `SafeTensors::deserialize` and `TensorView` behave exactly as they did
    /// over the whole shard. The only difference is how many bytes were read.
    pub fn single_tensor_image(
        &mut self,
        name: &str,
    ) -> Result<(Vec<u8>, TensorDtype, Vec<usize>)> {
        let entry = self.entry(name)?;
        let raw = self.read_payload_entry(&entry)?;
        self.ledger.whole_tensor_reads += 1;
        let escaped = name.replace('\\', "\\\\").replace('"', "\\\"");
        let dims = entry
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut header = format!(
            "{{\"{escaped}\":{{\"dtype\":\"{}\",\"shape\":[{dims}],\"data_offsets\":[0,{}]}}}}",
            entry.dtype.name(),
            raw.len()
        );
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut image = Vec::with_capacity(8 + header.len() + raw.len());
        image.extend_from_slice(&(header.len() as u64).to_le_bytes());
        image.extend_from_slice(header.as_bytes());
        image.extend_from_slice(&raw);
        Ok((image, entry.dtype, entry.shape))
    }

    fn read_payload_entry(&mut self, entry: &TensorEntry) -> Result<Vec<u8>> {
        let length = entry.byte_len()? as usize;
        // A single-row-at-a-time shard read would be slower than the previous
        // whole-file read, so a whole-tensor read is one bounded seek.
        self.read_shard_range(entry, 0, length)
    }

    /// Decode `F32`/`F16`/`BF16` payload bytes into `f32`, rejecting non-finite
    /// values exactly as the previous host reader did.
    pub fn decode_f32(dtype: TensorDtype, raw: &[u8]) -> Result<Vec<f32>> {
        let width = dtype.item_bytes();
        if raw.len() % width != 0 {
            bail!(
                "{} payload of {} bytes is not a whole number of {}-byte items",
                dtype.name(),
                raw.len(),
                width
            );
        }
        let values: Vec<f32> = match dtype {
            TensorDtype::F32 => f32_values_from_le_bytes(raw),
            TensorDtype::F16 => raw
                .chunks_exact(2)
                .map(|word| f16_bits_to_f32(u16::from_le_bytes([word[0], word[1]])))
                .collect(),
            TensorDtype::Bf16 => raw
                .chunks_exact(2)
                .map(|word| f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16))
                .collect(),
            other => bail!(
                "tensor storage {} cannot be decoded as f32; use the matching decode_* helper",
                other.name()
            ),
        };
        if values.iter().any(|value| !value.is_finite()) {
            bail!("decoded {:?} payload contains non-finite values", dtype);
        }
        Ok(values)
    }

    pub fn decode_f64(dtype: TensorDtype, raw: &[u8]) -> Result<Vec<f64>> {
        let values: Vec<f64> = match dtype {
            TensorDtype::F64 => raw
                .chunks_exact(8)
                .map(|word| {
                    f64::from_le_bytes([
                        word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7],
                    ])
                })
                .collect(),
            TensorDtype::F32 => raw
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]) as f64)
                .collect(),
            TensorDtype::F16 | TensorDtype::Bf16 => Self::decode_f32(dtype, raw)?
                .into_iter()
                .map(f64::from)
                .collect(),
            other => bail!(
                "tensor storage {} cannot be decoded as f64; use the matching decode_* helper",
                other.name()
            ),
        };
        if values.iter().any(|value| !value.is_finite()) {
            bail!("decoded {:?} payload contains non-finite values", dtype);
        }
        Ok(values)
    }

    /// Integer widening, matching the accepted storage set and message shape of
    /// the host reader that used to own this path: U8/U32/I32/I64 into `i64`.
    /// These are integer tensors, so no finiteness check applies.
    pub fn decode_i64(name: &str, dtype: TensorDtype, raw: &[u8]) -> Result<Vec<i64>> {
        let mut values = Vec::with_capacity(raw.len() / dtype.item_bytes().max(1));
        match dtype {
            TensorDtype::U8 => values.extend(raw.iter().copied().map(i64::from)),
            TensorDtype::U32 => {
                Self::require_whole_items(name, raw, 4)?;
                values.extend(
                    raw.chunks_exact(4)
                        .map(|word| i64::from(u32::from_le_bytes(word.try_into().unwrap()))),
                );
            }
            TensorDtype::I32 => {
                Self::require_whole_items(name, raw, 4)?;
                values.extend(
                    raw.chunks_exact(4)
                        .map(|word| i64::from(i32::from_le_bytes(word.try_into().unwrap()))),
                );
            }
            TensorDtype::I64 => {
                Self::require_whole_items(name, raw, 8)?;
                values.extend(
                    raw.chunks_exact(8)
                        .map(|word| i64::from_le_bytes(word.try_into().unwrap())),
                );
            }
            other => bail!(
                "integer tensor {name:?} must be U8/U32/I32/I64, got {}",
                other.name()
            ),
        }
        Ok(values)
    }

    /// Integer narrowing into `u32`, matching the host reader that used to own
    /// this path: U8/U32/I32/I64, with negative and out-of-range values rejected
    /// rather than wrapped.
    pub fn decode_u32(name: &str, dtype: TensorDtype, raw: &[u8]) -> Result<Vec<u32>> {
        match dtype {
            TensorDtype::U8 => Ok(raw.iter().copied().map(u32::from).collect()),
            TensorDtype::U32 => {
                Self::require_whole_items(name, raw, 4)?;
                Ok(raw
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect())
            }
            TensorDtype::I32 => {
                Self::require_whole_items(name, raw, 4)?;
                raw.chunks_exact(4)
                    .map(|word| {
                        let value = i32::from_le_bytes(word.try_into().unwrap());
                        u32::try_from(value).with_context(|| {
                            format!("token tensor {name:?} contains negative value {value}")
                        })
                    })
                    .collect()
            }
            TensorDtype::I64 => {
                Self::require_whole_items(name, raw, 8)?;
                raw.chunks_exact(8)
                    .map(|word| {
                        let value = i64::from_le_bytes(word.try_into().unwrap());
                        u32::try_from(value).with_context(|| {
                            format!("token tensor {name:?} contains out-of-range token {value}")
                        })
                    })
                    .collect()
            }
            other => bail!(
                "token tensor {name:?} must be U8/U32/I32/I64, got {}",
                other.name()
            ),
        }
    }

    fn require_whole_items(name: &str, raw: &[u8], width: usize) -> Result<()> {
        if raw.len() % width != 0 {
            bail!(
                "tensor {name:?} has invalid {width}-byte item byte length {}",
                raw.len()
            );
        }
        Ok(())
    }

    /// Convenience: shape plus decoded `f32` values for one tensor.
    pub fn read_f32(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (dtype, shape, raw) = self.read_payload(name)?;
        self.ledger.decoded_bytes = self.ledger.decoded_bytes.saturating_add(raw.len() as u64);
        self.ledger.max_f32_request_bytes = self.ledger.max_f32_request_bytes.max((raw.len() / dtype.item_bytes() * 4) as u64);
        Ok((shape, Self::decode_f32(dtype, &raw)?))
    }
}

/// Copy F32 checkpoint bits in bulk when native endian/alignment permit it.
/// Callers retain their own whole-item and finiteness validation. The fallback
/// handles unaligned byte slices and big-endian hosts without unsafe casts.
pub(crate) fn f32_values_from_le_bytes(raw: &[u8]) -> Vec<f32> {
    #[cfg(target_endian = "little")]
    if let Ok(values) = bytemuck::try_cast_slice::<u8, f32>(raw) {
        return values.to_vec();
    }
    raw.chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// IEEE-754 binary16 to binary32 widening, including subnormals and infinities.
///
/// `crate::checkpoint` retains its own historical copy for paths that are not
/// routed through this module; `checkpoint`'s tests assert that the two agree on
/// all 65,536 bit patterns, so the duplication cannot drift.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let leading = mantissa.leading_zeros() - 22;
                let normalised = (mantissa << (leading + 1)) & 0x03ff;
                let exponent = 127 - 15 - leading;
                sign | ((exponent as u32) << 23) | (normalised << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | (((u32::from(exponent) + 127 - 15) as u32) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

/// Which tier a parameter set is expected to occupy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightResidency {
    /// The whole set fits beside what is already resident, so it may be held
    /// entirely on the device.
    #[default]
    Resident,
    /// The set does not fit; it must be staged through host memory, reading
    /// pieces of the package as they are needed.
    Tiered,
}

impl WeightResidency {
    pub fn is_resident(self) -> bool {
        matches!(self, Self::Resident)
    }
}

/// Bytes the device could still take, or `None` when the driver will not say.
///
/// A missing budget is deliberately reported as unknown rather than as zero:
/// callers that cannot plan must fall back to bounded staging, and `None` makes
/// that explicit instead of silently looking like an empty device.
pub fn device_local_available_bytes(device: &VulkanDevice) -> Option<u64> {
    let budget = device.memory_budget().ok()?;
    if budget.device_local_available_bytes > 0 {
        return Some(budget.device_local_available_bytes);
    }
    budget
        .device_local_heap_size_bytes
        .checked_sub(budget.device_local_usage_bytes)
        .filter(|available| *available > 0)
}

/// Decide whether a parameter set may be held resident.
pub fn plan_weight_residency(
    device: &VulkanDevice,
    required_bytes: u64,
    reserve_bytes: u64,
) -> WeightResidency {
    plan_weight_residency_with_budget(
        device_local_available_bytes(device),
        required_bytes,
        reserve_bytes,
    )
}

/// Budget-injected form, so the decision is testable without a GPU.
pub fn plan_weight_residency_with_budget(
    available_bytes: Option<u64>,
    required_bytes: u64,
    reserve_bytes: u64,
) -> WeightResidency {
    let Some(available) = available_bytes else {
        return WeightResidency::Tiered;
    };
    let needed = required_bytes.saturating_add(reserve_bytes);
    if needed <= available {
        WeightResidency::Resident
    } else {
        WeightResidency::Tiered
    }
}

/// Guard that reports a ledger alongside the tier the caller actually used, so
/// downstream evidence can state the tier rather than assume it.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ResidencyReport {
    pub residency: WeightResidency,
    pub required_bytes: u64,
    pub reserve_bytes: u64,
    pub available_bytes: Option<u64>,
}

impl ResidencyReport {
    pub fn new(
        residency: WeightResidency,
        required_bytes: u64,
        reserve_bytes: u64,
        available_bytes: Option<u64>,
    ) -> Self {
        Self {
            residency,
            required_bytes,
            reserve_bytes,
            available_bytes,
        }
    }

    pub fn plan(
        device: &VulkanDevice,
        required_bytes: u64,
        reserve_bytes: u64,
    ) -> (Self, WeightResidency) {
        let available = device_local_available_bytes(device);
        let residency = plan_weight_residency_with_budget(available, required_bytes, reserve_bytes);
        (
            Self::new(residency, required_bytes, reserve_bytes, available),
            residency,
        )
    }
}

/// One open package per shard file, shared by every reader in the process.
///
/// A vocabulary-sized table is consumed by a gather on every step, and
/// re-opening the package per gather would throw away the host tile cache (and
/// restart its statistics) each time. The registry keeps one `ModelPackage` per
/// canonical path, so the byte budget and the cache-hit ledger belong to the
/// process, not to one call.
fn shared_packages() -> &'static Mutex<HashMap<PathBuf, Arc<Mutex<ModelPackage>>>> {
    static PACKAGES: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<ModelPackage>>>>> = OnceLock::new();
    PACKAGES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Open, or reuse, the shared package that owns `path`.
///
/// When `path` is the file a resolved tensor source points at, the package
/// addresses exactly that file and resolves tensors from its own header rather
/// than from a sibling index. That keeps the read independent of whether the
/// package is single-file or sharded, which is the same guarantee the
/// absolute-offset reader it replaces gave.
/// A directory instead owns one cache across the shards in its index.
///
/// Repeated calls return the same handle, so `Arc::ptr_eq` is a valid identity
/// check and the ledger accumulates across calls.
pub fn shared_package(path: impl AsRef<Path>) -> Result<Arc<Mutex<ModelPackage>>> {
    let path = path.as_ref();
    let key = path
        .canonicalize()
        .with_context(|| format!("opening shared weight package at {}", path.display()))?;
    let mut packages = shared_packages()
        .lock()
        .map_err(|_| anyhow::anyhow!("shared weight-offload package registry is poisoned"))?;
    if let Some(package) = packages.get(&key) {
        return Ok(package.clone());
    }
    let reader = if key.is_dir() {
        ModelPackage::open_directory(&key)?
    } else {
        ModelPackage::open_file(&key)?
    };
    let package = Arc::new(Mutex::new(reader));
    packages.insert(key, package.clone());
    Ok(package)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn flat_views_reuse_physical_tiles_without_row_geometry_collisions() -> Result<()> {
        let dir=scratch("flat-cache");
        let path=dir.join("model.safetensors");
        let values=(0..24).map(|n| n as f32 * 0.125).collect::<Vec<_>>();
        write_package(&path,&[("w",TensorDtype::F32,vec![4,4],f32_bytes(&values[..16])),
            ("odd",TensorDtype::F32,vec![3,3],f32_bytes(&values[..9])),
            ("packed",TensorDtype::F32,vec![2,3,4],f32_bytes(&values))]);
        let mut package=ModelPackage::open_file(&path)?.with_tile_bytes(32).with_cache_bytes(128);
        assert_eq!(package.read_f32_rows("w",1,1)?,values[4..8]);
        let reads=package.ledger().payload_reads;
        assert_eq!(package.read_f32_elements("w",4,4)?,values[4..8]);
        assert_eq!(package.ledger().payload_reads,reads,"equal physical ranges must share a cache entry");
        assert_eq!(package.read_f32_elements("w",8,2)?,values[8..10]);
        let reads=package.ledger().payload_reads;
        assert_eq!(package.read_f32_rows("w",2,2)?,values[8..16]);
        assert_eq!(package.ledger().payload_reads,reads);
        // Row-aligned tiles here span 24 bytes; flat-aligned tiles span 32.
        // Their equal start offset must not turn them into the same cache key.
        assert_eq!(package.read_f32_rows("odd",0,1)?,values[..3]);
        assert_eq!(package.read_f32_elements("odd",5,4)?,values[5..9]);
        assert_eq!(package.read_f32_elements("packed",7,11)?,values[7..18]);
        assert!(package.ledger().cache_peak_bytes<=128);
        assert!(package.read_f32_elements("packed",23,2).is_err());
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn flat_cache_eviction_disabled_reads_and_finite_checks_remain_bounded() -> Result<()> {
        let dir=scratch("flat-cache-errors");
        let path=dir.join("model.safetensors");
        let mut values=(0..24).map(|n| n as f32).collect::<Vec<_>>();
        values[23]=f32::NAN;
        write_package(&path,&[("w",TensorDtype::F32,vec![2,3,4],f32_bytes(&values))]);
        let mut package=ModelPackage::open_file(&path)?.with_tile_bytes(16).with_cache_bytes(32);
        assert_eq!(package.read_f32_elements("w",0,2)?,values[..2]);
        assert_eq!(package.read_f32_elements("w",2,2)?,values[2..4]);
        assert_eq!(package.ledger().payload_reads,1);
        for start in [8,16,0] {assert_eq!(package.read_f32_elements("w",start,1)?,values[start..start+1]);}
        assert!(package.ledger().tiles_evicted>=2);
        // The prefetched tile contains a NaN outside this request. Only the
        // requested elements are decoded/validated; the NaN itself must reject.
        assert_eq!(package.read_f32_elements("w",20,2)?,values[20..22]);
        assert!(package.read_f32_elements("w",23,1).is_err());
        assert!(package.ledger().cache_peak_bytes<=32);
        let mut uncached=ModelPackage::open_file(&path)?.with_tile_bytes(16).with_cache_bytes(0);
        for _ in 0..2 {assert_eq!(uncached.read_f32_elements("w",3,2)?,values[3..5]);}
        assert_eq!(uncached.ledger().payload_bytes_read,16);
        assert_eq!(uncached.ledger().cache_hits,0);
        assert_eq!(uncached.ledger().cache_peak_bytes,0);
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn bulk_f32_decode_preserves_bits_and_unaligned_fallback_errors() -> Result<()> {
        let bits=[0u32,0x80000000,1,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x7f7fffff];
        let raw=bits.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>();
        let mut unaligned=vec![0u8];unaligned.extend_from_slice(&raw);
        for bytes in [&raw[..],&unaligned[1..]] {
            let decoded=ModelPackage::decode_f32(TensorDtype::F32,bytes)?;
            assert_eq!(decoded.iter().map(|n|n.to_bits()).collect::<Vec<_>>(),bits);
        }
        for bad in [f32::INFINITY,f32::NAN,f32::NEG_INFINITY] {
            let mut bytes=vec![0u8];bytes.extend_from_slice(&bad.to_le_bytes());
            assert!(ModelPackage::decode_f32(TensorDtype::F32,&bytes[1..]).is_err());
        }
        assert!(ModelPackage::decode_f32(TensorDtype::F32,&raw[..raw.len()-1]).is_err());
        Ok(())
    }

    /// Minimal SafeTensors writer for tests. Mirrors the released format: an
    /// 8-byte little-endian header length, a JSON header padded to 8 bytes, then
    /// the payloads.
    fn write_package(path: &Path, tensors: &[(&str, TensorDtype, Vec<usize>, Vec<u8>)]) {
        fn json_escape(value: &str) -> String {
            value.replace('\\', "\\\\").replace('"', "\\\"")
        }
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (index, (name, dtype, shape, payload)) in tensors.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let dims = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                json_escape(name),
                dtype.name(),
                dims,
                offset,
                offset + payload.len()
            ));
            offset += payload.len();
        }
        header.push('}');
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for (_, _, _, payload) in tensors {
            bytes.extend_from_slice(payload);
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// A test package directory that removes itself when the test ends, on the
    /// panic path as well as the success path.
    ///
    /// These fixtures are real SafeTensors packages, so a test that leaves one
    /// behind is storage the next run has to find and reclaim. The guard makes
    /// the cleanup a property of holding the directory rather than a line each
    /// test has to remember to reach.
    struct Scratch {
        path: PathBuf,
    }

    impl std::ops::Deref for Scratch {
        type Target = Path;

        fn deref(&self) -> &Path {
            &self.path
        }
    }

    impl AsRef<Path> for Scratch {
        fn as_ref(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn scratch(name: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "hierarchos-weight-offload-{}-{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Scratch { path }
    }

    #[test]
    fn logical_row_prefix_materializes_only_addressable_rows() -> Result<()> {
        let root = scratch("logical-prefix");
        let path = root.join("model.safetensors");
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, f32::NAN, f32::INFINITY];
        write_package(
            &path,
            &[("w", TensorDtype::F32, vec![4, 2], f32_bytes(&values))],
        );
        let mut package = ModelPackage::open_file(&path)?
            .with_tile_bytes(8)
            .with_cache_bytes(16);
        assert_eq!(package.read_f32_row_prefix("w", 3)?, values[..6]);
        assert_eq!(package.read_f32_row_prefix("w", 0)?, Vec::<f32>::new());
        assert!(package.read_f32_row_prefix("w", 5).is_err());
        assert!(
            package.read_all_f32_rows("w").is_err(),
            "full materialization must still validate the tail"
        );
        assert!(package.ledger().cache_peak_bytes <= 16);
        Ok(())
    }

    #[test]
    fn bounded_read_decodes_exactly_the_previous_whole_file_values() -> Result<()> {
        let root = scratch("values");
        let rows = 64usize;
        let cols = 8usize;
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| (index as f32 - 100.0) * 0.25)
            .collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[("w", TensorDtype::F32, vec![rows, cols], f32_bytes(&values))],
        );

        let mut package = ModelPackage::open_file(&path)?;
        assert_eq!(package.shape("w")?, vec![rows, cols]);
        assert_eq!(package.dtype("w")?, TensorDtype::F32);
        assert_eq!(package.tensor_bytes("w")?, (rows * cols * 4) as u64);
        let (shape, decoded) = package.read_f32("w")?;
        assert_eq!(shape, vec![rows, cols]);
        assert_eq!(decoded, values);
        Ok(())
    }

    #[test]
    fn row_reads_touch_only_the_requested_bytes_and_reuse_host_tiles() -> Result<()> {
        let root = scratch("rows");
        let rows = 4096usize;
        let cols = 64usize;
        let values: Vec<f32> = (0..rows * cols).map(|index| index as f32).collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[("w", TensorDtype::F32, vec![rows, cols], f32_bytes(&values))],
        );

        // One tile row is 256 bytes, so an 8 KiB tile holds 32 rows.
        let mut package = ModelPackage::open_file(&path)?
            .with_tile_bytes(8192)
            .with_cache_bytes(1024 * 1024);
        let payload = package.tensor_bytes("w")?;

        let first = package.read_rows("w", 4, 3)?;
        let expected: Vec<u8> = f32_bytes(&values[4 * cols..7 * cols]);
        assert_eq!(first, expected);
        let after_first = package.ledger();
        // One 32-row tile = 8192 bytes, far below the whole 1 MiB tensor.
        assert_eq!(after_first.payload_bytes_read, 8192);
        assert!(after_first.payload_bytes_read < payload);
        assert_eq!(after_first.cache_misses, 1);

        // A second read inside the same tile is served from host memory.
        let second = package.read_rows("w", 6, 2)?;
        assert_eq!(second, f32_bytes(&values[6 * cols..8 * cols]));
        let after_second = package.ledger();
        assert_eq!(after_second.cache_hits, 1);
        assert_eq!(after_second.payload_bytes_read, 8192);
        assert_eq!(after_second.payload_bytes_from_cache, 8192);

        // A read spanning two tiles pulls exactly the second tile.
        let spanning = package.read_rows("w", 30, 4)?;
        assert_eq!(spanning, f32_bytes(&values[30 * cols..34 * cols]));
        let after_span = package.ledger();
        assert_eq!(after_span.cache_misses, 2);
        assert_eq!(after_span.payload_bytes_read, 16384);
        assert!(after_span.cache_hit_ratio() > 0.0);
        Ok(())
    }

    /// A gather is the shape a vocabulary-sized table is consumed in. The
    /// previous reader for this consumer opened the shard and seeked once per
    /// wanted row; a tile-aligned gather pulls each host tile once for the whole
    /// request, keeps caller order, and serves a repeated row from the one read.
    #[test]
    fn row_gather_pulls_each_wanted_tile_once_and_preserves_caller_order() -> Result<()> {
        let root = scratch("gather");
        let rows = 4096usize;
        let cols = 64usize;
        let values: Vec<f32> = (0..rows * cols).map(|index| index as f32).collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[("w", TensorDtype::F32, vec![rows, cols], f32_bytes(&values))],
        );
        let row = |index: usize| f32_bytes(&values[index * cols..(index + 1) * cols]);

        // 8 KiB tiles of 32 rows each: rows 100, 5 and 33 live in tiles 3, 0 and 1.
        let mut package = ModelPackage::open_file(&path)?
            .with_tile_bytes(8192)
            .with_cache_bytes(1024 * 1024);
        let gathered = package.read_row_gather("w", &[100, 5, 100, 33])?;
        let mut expected = Vec::new();
        for index in [100usize, 5, 100, 33] {
            expected.extend_from_slice(&row(index));
        }
        assert_eq!(
            gathered, expected,
            "gather must keep caller order and duplicates"
        );
        let first = package.ledger();
        assert_eq!(first.row_gathers, 1);
        assert_eq!(first.cache_misses, 3, "three distinct tiles, not four rows");
        assert_eq!(first.payload_reads, 3);
        assert_eq!(first.payload_bytes_read, 3 * 8192);
        assert_eq!(first.rows_read, 96);

        // A second gather that lands in an already-resident tile reads nothing.
        let again = package.read_row_gather("w", &[101, 99])?;
        assert_eq!(again, [row(101), row(99)].concat());
        let second = package.ledger();
        assert_eq!(second.cache_hits, 1);
        assert_eq!(second.payload_reads, 3);
        assert_eq!(second.payload_bytes_read, 3 * 8192);
        assert!(second.payload_bytes_from_cache > 0);
        assert_eq!(second.row_gathers, 2);

        // The whole table is far larger than what a gather ever materialises:
        // three 8 KiB tiles against a 1 MiB table.
        let table = package.tensor_bytes("w")?;
        assert!(second.payload_bytes_read * 40 < table);
        Ok(())
    }

    #[test]
    fn row_gather_rejects_out_of_range_rows_and_rank_1_tensors() -> Result<()> {
        let root = scratch("gather-range");
        let values: Vec<f32> = (0..20).map(|index| index as f32).collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[
                ("w", TensorDtype::F32, vec![4, 5], f32_bytes(&values)),
                ("v", TensorDtype::F32, vec![4], f32_bytes(&values[..4])),
            ],
        );
        let mut package = ModelPackage::open_file(&path)?;
        let error = package
            .read_row_gather("w", &[4])
            .expect_err("a gather past the row count must fail closed");
        assert!(
            error.to_string().contains("outside w with 4 rows"),
            "{error}"
        );
        assert!(package.read_row_gather("v", &[0]).is_err());
        assert_eq!(package.read_row_gather("w", &[])?, Vec::<u8>::new());
        Ok(())
    }

    /// The heaviest consumer in the tree is a vocabulary-sized table read by a
    /// gather on every step. One shared package per file is what lets the second
    /// and later gathers reuse the host tier instead of starting cold.
    #[test]
    fn shared_packages_are_reused_across_callers_and_one_ledger() -> Result<()> {
        let root = scratch("shared");
        let rows = 512usize;
        let cols = 16usize;
        let values: Vec<f32> = (0..rows * cols).map(|index| index as f32).collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[("w", TensorDtype::F32, vec![rows, cols], f32_bytes(&values))],
        );

        let registered = shared_packages().lock().unwrap().len();
        let first = shared_package(&path)?;
        let second = shared_package(&path)?;
        assert!(
            Arc::ptr_eq(&first, &second),
            "the registry must reuse the handle"
        );
        // Other tests in this binary may hold their own entries, so assert the
        // registry grew by exactly this file rather than that it is size one.
        assert_eq!(shared_packages().lock().unwrap().len(), registered + 1);

        let gather = |handle: &Arc<Mutex<ModelPackage>>| -> Result<Vec<u8>> {
            let mut package = handle
                .lock()
                .map_err(|_| anyhow::anyhow!("shared package lock failed"))?;
            package.read_row_gather("w", &[7, 9])
        };
        let expected = [&values[7 * cols..8 * cols], &values[9 * cols..10 * cols]].concat();
        assert_eq!(gather(&first)?, f32_bytes(&expected));
        // A different caller reaching the same file through its own shard path
        // sees the same values and adds to the same ledger.
        assert_eq!(gather(&second)?, f32_bytes(&expected));
        let ledger = first.lock().unwrap().ledger();
        assert_eq!(ledger.row_gathers, 2);
        assert_eq!(ledger.headers_parsed, 1, "one header for the whole process");
        assert_eq!(
            ledger.cache_hits, 1,
            "the second gather is served from host"
        );
        assert!(first.lock().unwrap().root().is_dir());
        Ok(())
    }

    /// A vocabulary projection tile reads a bounded slice of the head and
    /// materializing the whole table walks the same bounded tiles.
    #[test]
    fn f32_row_tiles_and_full_materialization_read_through_the_cache() -> Result<()> {
        let root = scratch("f32rows");
        let rows = 4096usize;
        let cols = 64usize;
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| (index as f32 - 8192.0) * 0.125)
            .collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[("w", TensorDtype::F32, vec![rows, cols], f32_bytes(&values))],
        );

        // 8 KiB tiles are 32 rows, so one tile is one 32x64 slice.
        let mut package = ModelPackage::open_file(&path)?
            .with_tile_bytes(8192)
            .with_cache_bytes(1024 * 1024);
        let tile = package.read_f32_rows("w", 64, 32)?;
        assert_eq!(tile, values[64 * cols..96 * cols].to_vec());
        let ledger = package.ledger();
        assert_eq!(ledger.payload_bytes_read, 8192);
        assert_eq!(ledger.decoded_bytes, 8192);
        assert_eq!(ledger.cache_misses, 1);

        // Materializing the table walks it tile by tile under the same budget.
        let whole = package.read_all_f32_rows("w")?;
        assert_eq!(whole, values);
        let ledger = package.ledger();
        // 128 tiles, one of which the first read already made resident, so the
        // walk issues 127 reads and the package is read exactly once in total.
        assert_eq!(ledger.payload_reads, rows as u64 / 32);
        assert_eq!(ledger.cache_hits, 1);
        assert_eq!(ledger.payload_bytes_read, (rows * cols * 4) as u64);
        assert!(
            ledger.cache_peak_bytes <= 1024 * 1024,
            "materialization must stay inside the host budget"
        );
        assert!(
            ledger.payload_bytes_from_cache > 0,
            "tile 2 was already resident"
        );

        // A non-finite payload still fails closed on the tile path.
        let bad = scratch("f32rows-bad");
        let bad_path = bad.join("model.safetensors");
        write_package(
            &bad_path,
            &[(
                "w",
                TensorDtype::F32,
                vec![2, 2],
                f32_bytes(&[0.0, f32::INFINITY, 0.0, 0.0]),
            )],
        );
        assert!(ModelPackage::open_file(&bad_path)?
            .read_f32_rows("w", 0, 1)
            .is_err());
        Ok(())
    }

    #[test]
    fn host_cache_honours_its_byte_budget_and_evicts() -> Result<()> {
        let root = scratch("budget");
        let rows = 4096usize;
        let cols = 64usize;
        let values: Vec<f32> = (0..rows * cols).map(|index| index as f32).collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[("w", TensorDtype::F32, vec![rows, cols], f32_bytes(&values))],
        );

        // 8 KiB tiles, a 16 KiB budget: two tiles fit, a third must evict one.
        let mut package = ModelPackage::open_file(&path)?
            .with_tile_bytes(8192)
            .with_cache_bytes(16384);
        for tile in 0..3 {
            let first_row = tile * 32;
            let read = package.read_rows("w", first_row, 1)?;
            assert_eq!(
                read,
                f32_bytes(&values[first_row * cols..(first_row + 1) * cols])
            );
        }
        let ledger = package.ledger();
        assert_eq!(ledger.cache_misses, 3);
        assert_eq!(ledger.payload_bytes_read, 24576);
        assert!(
            ledger.cache_peak_bytes <= 16384,
            "cache exceeded its budget"
        );
        assert_eq!(ledger.tiles_evicted, 1);

        // With retention disabled, no read is ever served from the host tier.
        let mut uncached = ModelPackage::open_file(&path)?
            .with_tile_bytes(8192)
            .with_cache_bytes(0);
        uncached.read_rows("w", 0, 1)?;
        uncached.read_rows("w", 0, 1)?;
        let ledger = uncached.ledger();
        assert_eq!(ledger.cache_hits, 0);
        assert_eq!(ledger.payload_bytes_from_cache, 0);
        assert_eq!(ledger.cache_peak_bytes, 0);
        Ok(())
    }

    /// The cost of loading a package must scale with the tensors asked for, not
    /// with the package. The readers that every non-V4.1 architecture loads
    /// through previously called `fs::read` on the whole shard once per tensor,
    /// so a fifty-tensor shard was read and parsed fifty times.
    #[test]
    fn loading_many_tensors_reads_each_payload_once_and_parses_one_header() -> Result<()> {
        let root = scratch("many");
        let cols = 512usize;
        let tensor_count = 50usize;
        let mut tensors: Vec<(String, TensorDtype, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut expected: Vec<Vec<f32>> = Vec::new();
        for index in 0..tensor_count {
            let values: Vec<f32> = (0..cols)
                .map(|col| (index * cols + col) as f32 * 0.5)
                .collect();
            tensors.push((
                format!("layer.{index}.weight"),
                TensorDtype::F32,
                vec![1, cols],
                f32_bytes(&values),
            ));
            expected.push(values);
        }
        let path = root.join("model-00001-of-00001.safetensors");
        let borrowed: Vec<(&str, TensorDtype, Vec<usize>, Vec<u8>)> = tensors
            .iter()
            .map(|(name, dtype, shape, payload)| {
                (name.as_str(), *dtype, shape.clone(), payload.clone())
            })
            .collect();
        write_package(&path, &borrowed);
        let package_bytes = std::fs::metadata(&path)?.len();

        let mut package = ModelPackage::open_file(&path)?;
        for (index, want) in expected.iter().enumerate() {
            let name = format!("layer.{index}.weight");
            assert_eq!(&package.read_f32(&name)?.1, want);
        }
        let ledger = package.ledger();
        assert_eq!(ledger.payload_reads as usize, tensor_count);
        // Exactly the payload bytes, with the shard header parsed once instead
        // of once per tensor.
        assert_eq!(
            ledger.payload_bytes_read,
            package_bytes - 8 - {
                // The header length varies with the padding this writer chose, so
                // derive it from the image instead of hard-coding a constant.
                let image = std::fs::read(&path)?;
                u64::from_le_bytes(image[..8].try_into().unwrap())
            }
        );
        assert_eq!(ledger.headers_parsed, 1);
        let old_cost = package_bytes * tensor_count as u64;
        assert!(ledger.payload_bytes_read < old_cost / 40);
        Ok(())
    }

    #[test]
    fn sharded_packages_resolve_through_the_weight_map() -> Result<()> {
        let root = scratch("shards");
        let left: Vec<f32> = (0..32).map(|index| index as f32).collect();
        let right: Vec<f32> = (0..32).map(|index| index as f32 + 1000.0).collect();
        write_package(
            &root.join("model-00001-of-00002.safetensors"),
            &[("a.weight", TensorDtype::F32, vec![4, 8], f32_bytes(&left))],
        );
        write_package(
            &root.join("model-00002-of-00002.safetensors"),
            &[("b.weight", TensorDtype::F32, vec![4, 8], f32_bytes(&right))],
        );
        std::fs::write(
            root.join("model.safetensors.index.json"),
            br#"{"metadata":{"total_size":256},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#,
        )?;

        let mut package = ModelPackage::open_directory(&root)?;
        assert!(package.contains("a.weight")?);
        assert!(package.contains("b.weight")?);
        assert!(!package.contains("c.weight")?);
        assert_eq!(package.stored_bytes()?, 256);
        assert_eq!(package.read_f32("a.weight")?.1, left);
        assert_eq!(package.read_f32("b.weight")?.1, right);
        assert!(package.read_f32("c.weight").is_err());
        let shared=shared_package(&root)?;
        assert!(Arc::ptr_eq(&shared,&shared_package(&root)?));
        let mut reader=shared.lock().unwrap();
        assert_eq!(reader.read_f32_elements("a.weight",0,32)?,left);
        assert_eq!(reader.read_f32_elements("b.weight",0,32)?,right);
        let reads=reader.ledger().payload_reads;
        assert_eq!(reader.read_f32_elements("a.weight",0,32)?,left);
        assert_eq!(reader.ledger().payload_reads,reads,"one reader/cache must serve both shards");
        assert_eq!(reader.ledger().cache_peak_bytes,256);
        Ok(())
    }

    #[test]
    fn bf16_and_f16_decode_matches_their_f32_expansions() -> Result<()> {
        let root = scratch("half");
        // Every value here is exactly representable in both formats, so a
        // decode that rounds or truncates differently is caught.
        let values: Vec<f32> = vec![0.0, 1.0, -2.5, 0.5, 256.0, -0.0009765625];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let f16: Vec<u8> = values
            .iter()
            .flat_map(|value| f32_to_f16_bits(*value).to_le_bytes())
            .collect();
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[
                ("bf", TensorDtype::Bf16, vec![values.len()], bf16),
                ("hf", TensorDtype::F16, vec![values.len()], f16),
            ],
        );
        let mut package = ModelPackage::open_file(&path)?;
        let bf = package.read_f32("bf")?.1;
        let hf = package.read_f32("hf")?.1;
        for (index, value) in values.iter().enumerate() {
            assert_eq!(bf[index], *value, "BF16 value {index}");
            assert_eq!(hf[index], *value, "FP16 value {index}");
        }

        // The BF16 contract this crate's loaders already rely on is a *bit*
        // widening (high 16 bits, low 16 zeroed), not round-to-nearest. A value
        // that is not BF16-representable must therefore decode by truncation.
        // 65504 loses its low mantissa bits; 65536 is exact and must not move.
        assert_eq!(
            ModelPackage::decode_f32(
                TensorDtype::Bf16,
                &((65504.0f32.to_bits() >> 16) as u16).to_le_bytes()
            )?,
            vec![65280.0]
        );
        assert_eq!(
            ModelPackage::decode_f32(
                TensorDtype::Bf16,
                &((65536.0f32.to_bits() >> 16) as u16).to_le_bytes()
            )?,
            vec![65536.0]
        );
        Ok(())
    }

    fn f32_to_f16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mantissa = bits & 0x007f_ffff;
        if exponent <= 0 {
            return sign;
        }
        if exponent >= 0x1f {
            return sign | 0x7c00;
        }
        sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
    }

    #[test]
    fn non_finite_and_unsupported_payloads_fail_closed() -> Result<()> {
        let root = scratch("reject");
        let path = root.join("model.safetensors");
        write_package(
            &path,
            &[
                ("nan", TensorDtype::F32, vec![1], f32_bytes(&[f32::NAN])),
                ("u8", TensorDtype::U8, vec![4], vec![1, 2, 3, 4]),
            ],
        );
        let mut package = ModelPackage::open_file(&path)?;
        assert!(package.read_f32("nan").is_err());
        assert!(package.read_f32("u8").is_err());
        assert_eq!(
            ModelPackage::decode_u32("t", TensorDtype::U8, &[7, 8])?,
            vec![7, 8]
        );
        // Narrowing must fail closed rather than wrap.
        assert!(ModelPackage::decode_u32("t", TensorDtype::I32, &(-1i32).to_le_bytes()).is_err());
        assert!(ModelPackage::decode_u32(
            "t",
            TensorDtype::I64,
            &(u32::MAX as i64 + 1).to_le_bytes()
        )
        .is_err());
        assert_eq!(
            ModelPackage::decode_i64("t", TensorDtype::I32, &(-7i32).to_le_bytes())?,
            vec![-7]
        );
        Ok(())
    }

    #[test]
    fn residency_plans_fall_back_to_the_tiered_path_without_a_budget() {
        assert_eq!(
            plan_weight_residency_with_budget(None, 1, 0),
            WeightResidency::Tiered
        );
        assert_eq!(
            plan_weight_residency_with_budget(Some(1000), 600, 400),
            WeightResidency::Resident
        );
        assert_eq!(
            plan_weight_residency_with_budget(Some(1000), 601, 400),
            WeightResidency::Tiered
        );
        // Saturation must not wrap an overflowing request into "fits": a
        // requirement that saturates past the reported budget stays tiered.
        assert_eq!(
            plan_weight_residency_with_budget(Some(u64::MAX - 1), u64::MAX, 1),
            WeightResidency::Tiered
        );
        assert_eq!(
            plan_weight_residency_with_budget(Some(u64::MAX), u64::MAX, 1),
            WeightResidency::Resident
        );
    }
}
