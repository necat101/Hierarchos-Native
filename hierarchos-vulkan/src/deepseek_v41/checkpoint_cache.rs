//! Bounded immutable payload reuse, shared across one checkpoint's shards.
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub const DEFAULT_BYTES: usize = 32 * 1024 * 1024;
// Tiny scale tensors and gathered Engram rows consume little payload space.
// Bound their metadata independently without evicting an ordinary working set
// while most of the byte budget is still unused.
const MAX_ENTRIES: usize = 32768;
pub(crate) type CacheHandle = Arc<Mutex<PayloadCache>>;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct DeepseekV41CheckpointIoStats {
    pub payload_reads: u64,
    pub payload_bytes_read: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes: usize,
    pub cache_peak_bytes: usize,
    pub cache_entries: usize,
    pub evictions: u64,
    pub invalidations: u64,
    pub capacity_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct Key {
    pub path: PathBuf,
    pub offset: u64,
    pub len: usize,
    pub version: (u64, i64),
}

pub(crate) struct PayloadCache {
    stats: DeepseekV41CheckpointIoStats,
    clock: u64,
    versions: BTreeMap<PathBuf, (u64, i64)>,
    entries: BTreeMap<Key, (Arc<Vec<u8>>, u64)>,
    order: BTreeSet<(u64, Key)>,
}
impl Default for PayloadCache {
    fn default() -> Self {
        Self {
            stats: Default::default(),
            clock: 0,
            versions: BTreeMap::new(),
            entries: BTreeMap::new(),
            order: BTreeSet::new(),
        }
    }
}
impl PayloadCache {
    pub fn enable(&mut self) {
        self.stats.capacity_bytes = DEFAULT_BYTES;
    }
    pub fn stats(&self) -> DeepseekV41CheckpointIoStats {
        DeepseekV41CheckpointIoStats {
            cache_entries: self.entries.len(),
            ..self.stats
        }
    }
    fn remove(&mut self, key: &Key) {
        if let Some((bytes, age)) = self.entries.remove(key) {
            self.stats.cache_bytes -= bytes.len();
            self.order.remove(&(age, key.clone()));
        }
    }
    pub fn get(&mut self, key: &Key) -> Option<Arc<Vec<u8>>> {
        if self
            .versions
            .insert(key.path.clone(), key.version)
            .is_some_and(|old| old != key.version)
        {
            let stale = self
                .entries
                .keys()
                .filter(|k| k.path == key.path)
                .cloned()
                .collect::<Vec<_>>();
            for k in stale {
                self.remove(&k);
            }
            self.stats.invalidations += 1;
        }
        if let Some((bytes, age)) = self.entries.get_mut(key) {
            self.order.remove(&(*age, key.clone()));
            self.clock += 1;
            *age = self.clock;
            self.order.insert((*age, key.clone()));
            self.stats.cache_hits += 1;
            return Some(Arc::clone(bytes));
        }
        self.stats.cache_misses += 1;
        None
    }
    pub fn insert(&mut self, key: Key, bytes: &[u8]) {
        self.stats.payload_reads += 1;
        self.stats.payload_bytes_read += bytes.len() as u64;
        if bytes.len() > self.stats.capacity_bytes
            || self.stats.capacity_bytes == 0
            || self.versions.get(&key.path) != Some(&key.version)
        {
            return;
        }
        self.remove(&key);
        while self.stats.cache_bytes + bytes.len() > self.stats.capacity_bytes
            || self.entries.len() >= MAX_ENTRIES
        {
            let Some((_, victim)) = self.order.first().cloned() else {
                break;
            };
            self.remove(&victim);
            self.stats.evictions += 1;
        }
        self.clock += 1;
        self.stats.cache_bytes += bytes.len();
        self.stats.cache_peak_bytes = self.stats.cache_peak_bytes.max(self.stats.cache_bytes);
        self.order.insert((self.clock, key.clone()));
        self.entries
            .insert(key, (Arc::new(bytes.to_vec()), self.clock));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(n: usize, version: i64) -> Key {
        Key {
            path: PathBuf::from(format!("shard-{}", n % 2)),
            offset: n as u64,
            len: 8,
            version: (100, version),
        }
    }
    #[test]
    fn payload_cache_bounds_lru_and_invalidates_each_shard() {
        let mut cache = PayloadCache::default();
        cache.enable();
        cache.stats.capacity_bytes = 16;
        let a = key(0, 1);
        let b = key(1, 1);
        let c = key(2, 1);
        for k in [&a, &b] {
            assert!(cache.get(k).is_none());
            cache.insert(k.clone(), &[7; 8]);
        }
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&c).is_none());
        cache.insert(c.clone(), &[9; 8]);
        assert!(cache.get(&b).is_none());
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&key(0, 2)).is_none());
        assert_eq!(
            cache.stats().cache_bytes,
            0,
            "both cached ranges in changed shard must invalidate"
        );
        assert_eq!(cache.stats().cache_peak_bytes, 16);
        cache.insert(key(0, 2), &[0; 32]);
        assert_eq!(cache.stats().cache_bytes, 0);
    }

    #[test]
    fn payload_cache_metadata_count_is_bounded_too() {
        let mut cache = PayloadCache::default();
        cache.enable();
        for n in 0..=MAX_ENTRIES {
            let k = key(n, 1);
            assert!(cache.get(&k).is_none());
            cache.insert(k, &[0; 8]);
        }
        assert_eq!(cache.stats().cache_entries, MAX_ENTRIES);
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.stats().cache_peak_bytes <= DEFAULT_BYTES);
    }
}
