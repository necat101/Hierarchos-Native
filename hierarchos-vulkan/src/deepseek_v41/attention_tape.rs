//! Retained sparse attention and cache-write graph. Cache IDs name physical
//! sources, so shared readers and later decode steps join the same adjoint.
//! Sources are already rotated/quantized tensors; this assigns no derivative
//! to a quantizer, index selection, compressor or projection.

use super::*;
use crate::deepseek_v41::DeepseekV41AttentionAdjoints;
use std::sync::Arc;

#[cfg(test)]
#[path = "attention_tape_tests.rs"]
mod tests;

macro_rules! handle {
    ($name:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            owner: Arc<()>,
            index: usize,
        }
    };
}
handle!(DeepseekV41AttentionCacheId);
handle!(DeepseekV41AttentionWriteId);
handle!(DeepseekV41AttentionReadId);

/// Prefill reads the complete chunk, even when only its suffix seeds the ring.
/// Decode reads the current ring. Both paths join at the seed write's source.
#[derive(Clone, Debug)]
pub enum DeepseekV41AttentionWindowView {
    Cache(DeepseekV41AttentionCacheId),
    Prefill(DeepseekV41AttentionWriteId),
}

pub struct DeepseekV41AttentionRead {
    pub id: DeepseekV41AttentionReadId,
    pub output: GpuBuffer,
}

enum Cache {
    Window(DeepseekV41WindowKvCache),
    Compressed(DeepseekV41CompressedKvCache),
}
impl Cache {
    fn shape(&self) -> (usize, usize, usize) {
        match self {
            Self::Window(c) => (c.batch(), c.window(), c.head_dim()),
            Self::Compressed(c) => (c.batch(), c.capacity(), c.head_dim()),
        }
    }
    fn values(&self) -> &GpuBuffer {
        match self {
            Self::Window(c) => c.values(),
            Self::Compressed(c) => c.values(),
        }
    }
    fn len(&self) -> usize {
        let (b, n, d) = self.shape();
        b * n * d
    }
}
struct Write {
    cache: usize,
    slots: Vec<i32>,
    // Only a prefill seed needs the untruncated source for direct chunk reads.
    prefill: Option<GpuBuffer>,
}
struct Read {
    query: GpuBuffer,
    window: GpuBuffer,
    compressed: Option<GpuBuffer>,
    wi: GpuBuffer,
    ci: GpuBuffer,
    sink: GpuBuffer,
    params: DeepseekV41SparseAttentionParams,
    heads: usize,
    dim: usize,
    window_owner: DeepseekV41AttentionWindowView,
    compressed_owner: Option<usize>,
}
enum Event {
    Write(usize),
    Read(usize),
}

/// One recorded sequence/chunk graph. All writes to registered caches while
/// recording MUST use this tape; out-of-band writes are outside the contract.
/// The tape holds allocation aliases to native caches, without copying their
/// payloads or borrowing the runtime, so that runtime can continue executing.
/// Backward consumes the tape, releasing each read snapshot as it is visited.
/// Snapshot bytes are exposed for the future model activation-budget planner.
pub struct DeepseekV41AttentionTape {
    device: VulkanDevice,
    owner: Arc<()>,
    caches: Vec<Cache>,
    writes: Vec<Write>,
    reads: Vec<Read>,
    events: Vec<Event>,
    retained_bytes: usize,
}

pub struct DeepseekV41AttentionTapeGradients {
    owner: Arc<()>,
    queries: Vec<Option<GpuBuffer>>,
    sources: Vec<Option<GpuBuffer>>,
    initial_caches: Vec<Option<GpuBuffer>>,
}
impl DeepseekV41AttentionTapeGradients {
    /// An unseeded read is absent, rather than an artificial zero-gradient use.
    pub fn query(&self, id: &DeepseekV41AttentionReadId) -> Result<Option<&GpuBuffer>> {
        check_id(&self.owner, &id.owner, id.index, self.queries.len())?;
        Ok(self.queries[id.index].as_ref())
    }
    /// Post-quantization source gradient, including direct prefill readers and
    /// every later cache reader up to the overwrite that ends a row's lifetime.
    pub fn source(&self, id: &DeepseekV41AttentionWriteId) -> Result<Option<&GpuBuffer>> {
        check_id(&self.owner, &id.owner, id.index, self.sources.len())?;
        Ok(self.sources[id.index].as_ref())
    }
    pub fn initial_cache(&self, id: &DeepseekV41AttentionCacheId) -> Result<Option<&GpuBuffer>> {
        check_id(&self.owner, &id.owner, id.index, self.initial_caches.len())?;
        Ok(self.initial_caches[id.index].as_ref())
    }
}
fn check_id(owner: &Arc<()>, other: &Arc<()>, index: usize, len: usize) -> Result<()> {
    if !Arc::ptr_eq(owner, other) || index >= len {
        bail!("V4.1 attention tape handle belongs to a different graph or is invalid");
    }
    Ok(())
}
fn size(shape: &[usize]) -> Result<usize> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
        .context("V4.1 attention tape zero/overflowing geometry")?;
    if n > u32::MAX as usize {
        bail!("V4.1 attention tape exceeds u32 addressing");
    }
    Ok(n)
}

impl DeepseekV41AttentionTape {
    pub fn new(device: VulkanDevice) -> Self {
        Self {
            device,
            owner: Arc::new(()),
            caches: Vec::new(),
            writes: Vec::new(),
            reads: Vec::new(),
            events: Vec::new(),
            retained_bytes: 0,
        }
    }
    /// Physical caches are registered once. Every consumer of a shared source
    /// uses its existing ID; alias registration is rejected to prevent splitting
    /// one cache's gradients across independent ownership records.
    pub fn register_window(
        &mut self,
        cache: &DeepseekV41WindowKvCache,
    ) -> Result<DeepseekV41AttentionCacheId> {
        self.register(Cache::Window(cache.tape_alias()?))
    }
    pub fn register_compressed(
        &mut self,
        cache: &DeepseekV41CompressedKvCache,
    ) -> Result<DeepseekV41AttentionCacheId> {
        self.register(Cache::Compressed(cache.tape_alias()?))
    }
    fn register(&mut self, cache: Cache) -> Result<DeepseekV41AttentionCacheId> {
        let (b, n, d) = cache.shape();
        self.check(cache.values(), size(&[b, n, d])?)?;
        if n > i32::MAX as usize
            || self
                .caches
                .iter()
                .any(|c| c.values().shares_allocation_with(cache.values()))
        {
            bail!("V4.1 attention tape duplicate cache or invalid capacity");
        }
        let id = DeepseekV41AttentionCacheId {
            owner: self.owner.clone(),
            index: self.caches.len(),
        };
        self.caches.push(cache);
        Ok(id)
    }
    fn cache(&self, id: &DeepseekV41AttentionCacheId) -> Result<&Cache> {
        check_id(&self.owner, &id.owner, id.index, self.caches.len())?;
        Ok(&self.caches[id.index])
    }
    fn check(&self, input: &GpuBuffer, len: usize) -> Result<()> {
        if !input.belongs_to_device(&self.device) || input.f32_capacity() < len {
            bail!("V4.1 attention tape buffer geometry or device mismatch");
        }
        Ok(())
    }
    fn snapshot(&self, input: &GpuBuffer, len: usize) -> Result<GpuBuffer> {
        self.check(input, len)?;
        input.duplicate_f32(len)
    }
    fn bytes_after(&self, elements: usize) -> Result<usize> {
        elements
            .checked_mul(4)
            .and_then(|n| n.checked_add(self.retained_bytes))
            .context("V4.1 attention tape snapshot byte count overflow")
    }
    /// Snapshot payload bytes, excluding caller-owned caches, allocator padding,
    /// metadata and backward scratch. This is not a peak device-memory bound.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
    pub fn read_count(&self) -> usize {
        self.reads.len()
    }
    pub fn write_count(&self) -> usize {
        self.writes.len()
    }

    fn record_write(
        &mut self,
        cache: usize,
        slots: Vec<i32>,
        prefill: Option<GpuBuffer>,
    ) -> DeepseekV41AttentionWriteId {
        let id = DeepseekV41AttentionWriteId {
            owner: self.owner.clone(),
            index: self.writes.len(),
        };
        self.writes.push(Write {
            cache,
            slots,
            prefill,
        });
        self.events.push(Event::Write(id.index));
        id
    }
    pub fn seed_window(
        &mut self,
        id: &DeepseekV41AttentionCacheId,
        source: &GpuBuffer,
        seq: usize,
    ) -> Result<DeepseekV41AttentionWriteId> {
        let Cache::Window(cache) = self.cache(id)? else {
            bail!("V4.1 attention tape expected a window cache");
        };
        let len = size(&[cache.batch(), seq, cache.head_dim()])?;
        if cache.head_dim().div_ceil(64) > 65_535
            || cache.batch() * cache.window() > 65_535
            || seq > u32::MAX as usize
        {
            bail!("V4.1 window tape seed exceeds dispatch capacity");
        }
        let bytes = self.bytes_after(len)?;
        let copy = self.snapshot(source, len)?;
        cache.seed_prefill(&copy, seq)?;
        let slots = (0..seq)
            .map(|row| {
                if row + cache.window() < seq {
                    -1
                } else {
                    (row % cache.window()) as i32
                }
            })
            .collect();
        self.retained_bytes = bytes;
        Ok(self.record_write(id.index, slots, Some(copy)))
    }
    pub fn write_window(
        &mut self,
        id: &DeepseekV41AttentionCacheId,
        source: &GpuBuffer,
        position: usize,
    ) -> Result<DeepseekV41AttentionWriteId> {
        let Cache::Window(cache) = self.cache(id)? else {
            bail!("V4.1 attention tape expected a window cache");
        };
        self.check(source, size(&[cache.batch(), cache.head_dim()])?)?;
        if cache.head_dim().div_ceil(64) > 65_535
            || cache.batch() * cache.window() > 65_535
            || position > u32::MAX as usize
            || source.shares_allocation_with(cache.values())
        {
            bail!("V4.1 window tape write exceeds dispatch capacity or aliases its cache");
        }
        cache.write_decode(source, position)?;
        let slot = (position % cache.window()) as i32;
        Ok(self.record_write(id.index, vec![slot], None))
    }
    pub fn append_compressed(
        &mut self,
        id: &DeepseekV41AttentionCacheId,
        source: &DeepseekV41QuantizedActivation,
        groups: usize,
        first_group: usize,
    ) -> Result<DeepseekV41AttentionWriteId> {
        let Cache::Compressed(cache) = self.cache(id)? else {
            bail!("V4.1 attention tape expected a compressed cache");
        };
        if groups == 0
            || first_group
                .checked_add(groups)
                .is_none_or(|end| end > cache.capacity())
            || source.values.shares_allocation_with(cache.values())
        {
            bail!("V4.1 compressed tape write range is invalid or aliases its cache");
        }
        size(&[cache.batch(), groups, cache.head_dim()])?;
        cache.append_groups(source, cache.batch(), groups, first_group)?;
        let slots = (first_group..first_group + groups)
            .map(|n| n as i32)
            .collect();
        Ok(self.record_write(id.index, slots, None))
    }

    /// Executes the unchanged native gathered forward, retaining independent
    /// GPU copies of Q, both KV sources, indices and the frozen sink. Releasing
    /// projection weights or mutating caller scratch after this call is safe.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &mut self,
        attention: &DeepseekV41Attention,
        query: &GpuBuffer,
        window: DeepseekV41AttentionWindowView,
        compressed: Option<&DeepseekV41AttentionCacheId>,
        wi: &GpuBuffer,
        ci: &GpuBuffer,
        params: DeepseekV41SparseAttentionParams,
    ) -> Result<DeepseekV41AttentionRead> {
        let p = params;
        let heads = attention.config.heads;
        let dim = attention.config.head_dim;
        if !attention.device.shares_logical_device_with(&self.device) {
            bail!("V4.1 attention tape forward device mismatch");
        }
        let q_len = size(&[p.batch, p.seq, heads, dim])?;
        let w_len = size(&[p.batch, p.window_rows, dim])?;
        if p.batch * p.seq > 65_535
            || heads > 65_535
            || p.window_topk == 0
            || p.window_topk > p.window_rows
            || p.compressed_topk > p.compressed_capacity
            || p.window_rows
                .checked_add(p.compressed_capacity)
                .is_none_or(|n| n > i32::MAX as usize)
        {
            bail!("V4.1 attention tape sparse geometry is invalid");
        }
        let slots = p
            .window_topk
            .checked_add(p.compressed_topk)
            .context("V4.1 attention tape slot overflow")?;
        size(&[p.batch, p.seq, heads, slots])?;
        size(&[p.batch, p.seq, slots, dim])?;
        size(&[p.batch, p.window_rows + p.compressed_capacity, dim])?;
        let w = match &window {
            DeepseekV41AttentionWindowView::Cache(id) => {
                let c = self.cache(id)?;
                if !matches!(c, Cache::Window(_)) || c.shape() != (p.batch, p.window_rows, dim) {
                    bail!("V4.1 attention tape window cache shape mismatch");
                }
                c.values()
            }
            DeepseekV41AttentionWindowView::Prefill(id) => {
                check_id(&self.owner, &id.owner, id.index, self.writes.len())?;
                let write = &self.writes[id.index];
                if self.caches[write.cache].shape().0 != p.batch
                    || self.caches[write.cache].shape().2 != dim
                    || write.slots.len() != p.window_rows
                    || p.seq != p.window_rows
                {
                    bail!("V4.1 attention tape prefill shape mismatch");
                }
                write
                    .prefill
                    .as_ref()
                    .context("V4.1 attention tape write is not a prefill seed")?
            }
        };
        let c = match compressed {
            Some(id) => {
                let c = self.cache(id)?;
                if !matches!(c, Cache::Compressed(_))
                    || c.shape() != (p.batch, p.compressed_capacity, dim)
                {
                    bail!("V4.1 attention tape compressed cache shape mismatch");
                }
                Some(c.values())
            }
            None => {
                if p.compressed_capacity != 0 || p.compressed_topk != 0 {
                    bail!("V4.1 attention tape missing compressed cache");
                }
                None
            }
        };
        let c_len = if c.is_some() {
            size(&[p.batch, p.compressed_capacity, dim])?
        } else {
            0
        };
        let wi_len = size(&[p.batch, p.seq, p.window_topk])?;
        let ci_len = p
            .batch
            .checked_mul(p.seq)
            .and_then(|n| n.checked_mul(p.compressed_topk))
            .context("V4.1 attention tape index size overflow")?;
        let elements = [q_len, w_len, c_len, wi_len, ci_len.max(1), heads]
            .into_iter()
            .try_fold(0usize, |a, b| a.checked_add(b))
            .context("V4.1 attention tape snapshot size overflow")?;
        let bytes = self.bytes_after(elements)?;
        let read = Read {
            query: self.snapshot(query, q_len)?,
            window: self.snapshot(w, w_len)?,
            compressed: c.map(|b| self.snapshot(b, c_len)).transpose()?,
            wi: self.snapshot(wi, wi_len)?,
            ci: if ci_len == 0 {
                GpuBuffer::zeros_u32(&self.device, 1)?
            } else {
                self.snapshot(ci, ci_len)?
            },
            sink: self.snapshot(attention.attention_sink()?, heads)?,
            params: p,
            heads,
            dim,
            window_owner: window,
            compressed_owner: compressed.map(|id| id.index),
        };
        let output = attention.gathered_attention(
            &read.query,
            &read.window,
            read.compressed.as_ref(),
            &read.wi,
            &read.ci,
            p,
        )?;
        let id = DeepseekV41AttentionReadId {
            owner: self.owner.clone(),
            index: self.reads.len(),
        };
        self.reads.push(read);
        self.events.push(Event::Read(id.index));
        self.retained_bytes = bytes;
        Ok(DeepseekV41AttentionRead { id, output })
    }

    /// Seeds are VJPs of the pre-rotation head output. Omitted reads are absent;
    /// explicit zero seeds remain real graph uses. All seeds are validated
    /// before dispatch. Reader joins use deterministic reverse event order.
    pub fn backward(
        self,
        seeds: &[(DeepseekV41AttentionReadId, GpuBuffer)],
    ) -> Result<DeepseekV41AttentionTapeGradients> {
        let mut upstream: Vec<Option<&GpuBuffer>> = (0..self.reads.len()).map(|_| None).collect();
        for (id, value) in seeds {
            check_id(&self.owner, &id.owner, id.index, self.reads.len())?;
            if upstream[id.index].is_some() {
                bail!("V4.1 attention tape duplicate backward seed");
            }
            let r = &self.reads[id.index];
            self.check(value, r.params.batch * r.params.seq * r.heads * r.dim)?;
            upstream[id.index] = Some(value);
        }
        let mut reverse = self.into_reverse()?;
        while let Some(id) = reverse.next_read()? {
            let seed = upstream[id.index];
            reverse.backward_read(&id, seed)?;
        }
        reverse.finish()
    }

    /// The model's reverse graph supplies each attention seed only after the
    /// following layers have been differentiated. This cursor preserves that
    /// topological order without waiting for every seed to exist in advance.
    pub fn into_reverse(mut self) -> Result<DeepseekV41AttentionReverseTape> {
        let reads = std::mem::take(&mut self.reads)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let result = DeepseekV41AttentionReverseTape {
            adjoint: DeepseekV41AttentionAdjoints::new(self.device.clone())?,
            join: Join::new(self.device.clone())?,
            queries: (0..reads.len()).map(|_| None).collect(),
            sources: (0..self.writes.len()).map(|_| None).collect(),
            cache_grad: (0..self.caches.len()).map(|_| None).collect(),
            write_done: vec![false; self.writes.len()],
            tape: self,
            reads,
        };
        Ok(result)
    }
}

pub struct DeepseekV41AttentionReverseTape {
    tape: DeepseekV41AttentionTape,
    reads: Vec<Option<Read>>,
    queries: Vec<Option<GpuBuffer>>,
    sources: Vec<Option<GpuBuffer>>,
    cache_grad: Vec<Option<GpuBuffer>>,
    write_done: Vec<bool>,
    adjoint: DeepseekV41AttentionAdjoints,
    join: Join,
}
impl DeepseekV41AttentionReverseTape {
    fn writes(&mut self) -> Result<()> {
        while let Some(Event::Write(i)) = self.tape.events.last() {
            let i = *i;
            let w = &self.tape.writes[i];
            if let Some(g) = self.cache_grad[w.cache].take() {
                let (batch, capacity, dim) = self.tape.caches[w.cache].shape();
                let result = self
                    .adjoint
                    .cache_write(&g, &w.slots, batch, capacity, dim)?;
                self.join.accumulate(
                    &mut self.sources[i],
                    result.grad_source,
                    batch * w.slots.len() * dim,
                )?;
                self.cache_grad[w.cache] = Some(result.grad_previous);
            }
            self.write_done[i] = true;
            self.tape.events.pop();
        }
        Ok(())
    }
    pub fn next_read(&mut self) -> Result<Option<DeepseekV41AttentionReadId>> {
        self.writes()?;
        Ok(match self.tape.events.last() {
            Some(Event::Read(i)) => Some(DeepseekV41AttentionReadId {
                owner: self.tape.owner.clone(),
                index: *i,
            }),
            _ => None,
        })
    }
    pub fn backward_read(
        &mut self,
        id: &DeepseekV41AttentionReadId,
        upstream: Option<&GpuBuffer>,
    ) -> Result<()> {
        check_id(&self.tape.owner, &id.owner, id.index, self.reads.len())?;
        if !matches!(self.tape.events.last(),Some(Event::Read(i)) if *i == id.index) {
            bail!("V4.1 reverse attention read is out of topological order");
        }
        let r = self.reads[id.index]
            .as_ref()
            .context("V4.1 attention read was already consumed")?;
        if let Some(g) = upstream {
            self.tape
                .check(g, r.params.batch * r.params.seq * r.heads * r.dim)?;
        }
        let r = self.reads[id.index].take().unwrap();
        if let Some(g) = upstream {
            let result = self.adjoint.sparse(
                &r.query,
                &r.window,
                r.compressed.as_ref(),
                &r.wi,
                &r.ci,
                &r.sink,
                g,
                r.heads,
                r.dim,
                r.params,
            )?;
            self.queries[id.index] = Some(result.grad_query);
            match r.window_owner {
                DeepseekV41AttentionWindowView::Cache(id) => self.join.accumulate(
                    &mut self.cache_grad[id.index],
                    result.grad_window,
                    self.tape.caches[id.index].len(),
                )?,
                DeepseekV41AttentionWindowView::Prefill(id) => self.join.accumulate(
                    &mut self.sources[id.index],
                    result.grad_window,
                    r.params.batch * r.params.window_rows * r.dim,
                )?,
            }
            if let (Some(id), Some(g)) = (r.compressed_owner, result.grad_compressed) {
                self.join
                    .accumulate(&mut self.cache_grad[id], g, self.tape.caches[id].len())?;
            }
        }
        self.tape.events.pop();
        self.writes()
    }
    pub fn take_query(&mut self, id: &DeepseekV41AttentionReadId) -> Result<Option<GpuBuffer>> {
        check_id(&self.tape.owner, &id.owner, id.index, self.reads.len())?;
        if self.reads[id.index].is_some() {
            bail!("V4.1 reverse attention query is not ready");
        }
        Ok(self.queries[id.index].take())
    }
    pub fn take_source(&mut self, id: &DeepseekV41AttentionWriteId) -> Result<Option<GpuBuffer>> {
        check_id(&self.tape.owner, &id.owner, id.index, self.sources.len())?;
        if !self.write_done[id.index] {
            bail!("V4.1 reverse attention source is not ready");
        }
        Ok(self.sources[id.index].take())
    }
    pub fn finish(mut self) -> Result<DeepseekV41AttentionTapeGradients> {
        self.writes()?;
        if !self.tape.events.is_empty() {
            bail!("V4.1 reverse attention graph is incomplete");
        }
        Ok(DeepseekV41AttentionTapeGradients {
            owner: self.tape.owner,
            queries: self.queries,
            sources: self.sources,
            initial_caches: self.cache_grad,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct JoinPush {
    len: u32,
    scale: f32,
    add: u32,
}
struct Join {
    device: VulkanDevice,
    kernel: vulkan::ComputeKernel,
}
impl Join {
    fn new(device: VulkanDevice) -> Result<Self> {
        use vulkan::BindingAccess::{ReadOnly as R, WriteOnly as W};
        let kernel = vulkan::ComputeKernel::new_with_access(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_projection_grad_vector.spv"),
            &[R, R, W],
            std::mem::size_of::<JoinPush>() as u32,
        )?;
        Ok(Self { device, kernel })
    }
    fn accumulate(
        &self,
        destination: &mut Option<GpuBuffer>,
        value: GpuBuffer,
        len: usize,
    ) -> Result<()> {
        if let Some(previous) = destination.as_ref() {
            let output = GpuBuffer::zeros_f32(&self.device, len)?;
            let groups = len.div_ceil(64);
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            self.kernel.record_dispatch(
                &mut commands,
                &[previous, &value, &output],
                bytemuck::bytes_of(&JoinPush {
                    len: len as u32,
                    scale: 1.0,
                    add: 1,
                }),
                [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1],
            )?;
            commands.submit()?;
            *destination = Some(output);
        } else {
            *destination = Some(value);
        }
        Ok(())
    }
}
