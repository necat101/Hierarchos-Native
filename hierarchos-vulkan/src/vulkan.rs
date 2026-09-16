use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::{CStr, CString},
    io::Cursor,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

use anyhow::{anyhow, bail, Context, Result};
use ash::{util::read_spv, vk, vk::Handle, Device, Entry, Instance};

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::CloseHandle;

include!(concat!(env!("OUT_DIR"), "/shader_debug_registry.rs"));

struct DeviceInner {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    timeline_semaphore_ext: Option<ash::khr::timeline_semaphore::Device>,
    queues: Vec<vk::Queue>,
    queue_locks: Vec<Mutex<()>>,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    command_pool_lock: Mutex<()>,
    host_memory_map_lock: Mutex<()>,
    command_buffer_ring: Mutex<CommandBufferRing>,
    submission_resource_arena: Mutex<TimelineSubmissionResourceArena>,
    recyclable_buffer_timeline_uses: Mutex<HashMap<u64, Vec<SubmissionTimelinePoint>>>,
    scratch_lease_timeline_uses: Mutex<HashMap<u64, Vec<SubmissionTimelinePoint>>>,
    scratch_buffer_arena: Mutex<ScratchBufferArena>,
    submission_timelines: Vec<Mutex<Option<QueueSubmissionTimeline>>>,
    submission_timeline_next_values: Vec<AtomicU64>,
    submission_timeline_enabled: bool,
    scheduler_kernel_timestamp_collection_enabled: AtomicBool,
    kernel_timestamp_profile_samples: AtomicU64,
    kernel_timestamp_profile_dispatches: AtomicU64,
    kernel_timestamp_profile_gpu_ns_total: AtomicU64,
    layout_interner: Mutex<LayoutInterner>,
    memory_allocator: Mutex<MemorySuballocator>,
    physical_device_index: usize,
    name: String,
    device_group_physical_device_count: u32,
    device_group_mask: u32,
    device_group_timeline_semaphore_enabled: bool,
    opaque_external_transport_enabled: bool,
    required_subgroup_size: Option<u32>,
    mixed_precision_capabilities: VulkanMixedPrecisionCapabilities,
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let (
                pending_resources,
                pending_buffer_allocations,
                reusable_descriptor_pools,
                reusable_buffer_allocations,
            ) = self
                .submission_resource_arena
                .get_mut()
                .map(|arena| {
                    (
                        std::mem::take(&mut arena.in_flight),
                        std::mem::take(&mut arena.in_flight_buffer_allocations),
                        std::mem::take(&mut arena.reusable_descriptor_pools),
                        std::mem::take(&mut arena.reusable_buffer_allocations),
                    )
                })
                .unwrap_or_default();
            for mut retirement in pending_resources {
                if let Some(profile) = retirement.kernel_timestamp_profile.take() {
                    for chunk in profile.chunks {
                        self.device.destroy_query_pool(chunk.pool, None);
                    }
                }
                for chunk in retirement.descriptor_pools.drain(..) {
                    self.device.destroy_descriptor_pool(chunk.pool, None);
                }
                for allocation in retirement.local_upload_allocations.drain(..) {
                    self.release_detached_buffer_allocation(allocation);
                }
            }
            for retirement in pending_buffer_allocations {
                self.release_detached_buffer_allocation(retirement.allocation);
            }
            for chunk in reusable_descriptor_pools {
                self.device.destroy_descriptor_pool(chunk.pool, None);
            }
            for allocation in reusable_buffer_allocations {
                self.release_detached_buffer_allocation(allocation);
            }
            for timeline in &mut self.submission_timelines {
                if let Ok(slot) = timeline.get_mut() {
                    if let Some(timeline) = slot.take() {
                        self.device.destroy_semaphore(timeline.semaphore, None);
                    }
                }
            }
            if let Ok(interner) = self.layout_interner.get_mut() {
                for pipeline_layout in interner.pipeline_layouts.values().copied() {
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                }
                for descriptor_layout in interner.descriptor_layouts.values().copied() {
                    self.device
                        .destroy_descriptor_set_layout(descriptor_layout, None);
                }
            }
            if let Ok(scratch) = self.scratch_buffer_arena.get_mut() {
                for slab in scratch.slabs.drain(..) {
                    self.device.destroy_buffer(slab.buffer, None);
                    if let Ok(allocator) = self.memory_allocator.get_mut() {
                        allocator.free(
                            &self.device,
                            slab.memory,
                            slab.memory_offset,
                            slab.allocation_span_bytes,
                            slab.capacity_bytes,
                        );
                    }
                }
            }
            if let Ok(allocator) = self.memory_allocator.get_mut() {
                allocator.release_all(&self.device);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
struct CommandBufferRing {
    reusable: VecDeque<vk::CommandBuffer>,
    allocated: u64,
    reused: u64,
    timeline_reaped: u64,
}

#[derive(Default)]
struct TimelineSubmissionResourceArena {
    in_flight: VecDeque<TimelineSubmissionRetirement>,
    in_flight_buffer_allocations: VecDeque<TimelineBufferRetirement>,
    in_flight_scratch_leases: VecDeque<TimelineScratchLeaseRetirement>,
    reusable_descriptor_pools: Vec<DescriptorPoolArenaChunk>,
    reusable_buffer_allocations: Vec<DetachedGpuBufferAllocation>,
    timeline_reaped: u64,
    timeline_retirement_latency_ns_total: u64,
    timeline_retirement_latency_ns_max: u64,
    timeline_retirement_latency_samples: u64,
    buffer_timeline_reaped: u64,
    scratch_timeline_reaped: u64,
    descriptor_pool_allocated: u64,
    descriptor_pool_reused: u64,
    buffer_allocation_reused: u64,
}

struct TimelineSubmissionRetirement {
    command_buffer: vk::CommandBuffer,
    descriptor_pools: Vec<DescriptorPoolArenaChunk>,
    local_upload_allocations: Vec<DetachedGpuBufferAllocation>,
    kernel_timestamp_profile: Option<KernelTimestampProfile>,
    semaphore: vk::Semaphore,
    value: u64,
    submitted_at: Instant,
}

#[derive(Clone, Copy)]
struct SubmissionTimelinePoint {
    semaphore: vk::Semaphore,
    value: u64,
}

struct TimelineBufferRetirement {
    allocation: DetachedGpuBufferAllocation,
    waits: Vec<SubmissionTimelinePoint>,
}

struct TimelineScratchLeaseRetirement {
    lease: ScratchLeaseToken,
    waits: Vec<SubmissionTimelinePoint>,
}

struct DetachedGpuBufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_offset: usize,
    allocation_span_bytes: usize,
    size_bytes: usize,
    memory_flags: vk::MemoryPropertyFlags,
    dedicated_memory: bool,
}

#[derive(Clone, Copy)]
struct QueueSubmissionTimeline {
    semaphore: vk::Semaphore,
}

#[derive(Clone, Copy)]
enum SubmissionCompletion {
    Fence(vk::Fence),
    Timeline {
        semaphore: vk::Semaphore,
        value: u64,
    },
}

impl DeviceInner {
    fn with_command_pool_host_access<T>(&self, op: impl FnOnce() -> T) -> Result<T> {
        let _guard = self
            .command_pool_lock
            .lock()
            .map_err(|_| anyhow!("Vulkan command-pool host-access lock was poisoned"))?;
        Ok(op())
    }

    fn reap_completed_submission_resources(&self) -> Result<usize> {
        let mut arena = self
            .submission_resource_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
        if arena.in_flight.is_empty()
            && arena.in_flight_buffer_allocations.is_empty()
            && arena.in_flight_scratch_leases.is_empty()
        {
            return Ok(0);
        }

        let mut observed_timelines = HashMap::<u64, u64>::new();
        let mut still_in_flight = VecDeque::with_capacity(arena.in_flight.len());
        let mut completed = Vec::new();
        while let Some(retirement) = arena.in_flight.pop_front() {
            let semaphore_key = retirement.semaphore.as_raw();
            let observed = if let Some(observed) = observed_timelines.get(&semaphore_key) {
                *observed
            } else {
                let observed = self.submission_timeline_counter_value(retirement.semaphore)?;
                observed_timelines.insert(semaphore_key, observed);
                observed
            };
            if observed >= retirement.value {
                completed.push(retirement);
            } else {
                still_in_flight.push_back(retirement);
            }
        }
        arena.in_flight = still_in_flight;
        arena.timeline_reaped = arena.timeline_reaped.saturating_add(completed.len() as u64);
        let observed_at = Instant::now();
        for retirement in &completed {
            let latency_ns = u64::try_from(
                observed_at
                    .saturating_duration_since(retirement.submitted_at)
                    .as_nanos(),
            )
            .unwrap_or(u64::MAX);
            arena.timeline_retirement_latency_ns_total = arena
                .timeline_retirement_latency_ns_total
                .saturating_add(latency_ns);
            arena.timeline_retirement_latency_ns_max =
                arena.timeline_retirement_latency_ns_max.max(latency_ns);
            arena.timeline_retirement_latency_samples =
                arena.timeline_retirement_latency_samples.saturating_add(1);
        }

        let mut still_in_flight_buffers =
            VecDeque::with_capacity(arena.in_flight_buffer_allocations.len());
        let mut completed_buffers = Vec::new();
        while let Some(retirement) = arena.in_flight_buffer_allocations.pop_front() {
            let mut complete = true;
            for point in &retirement.waits {
                let semaphore_key = point.semaphore.as_raw();
                let observed = if let Some(observed) = observed_timelines.get(&semaphore_key) {
                    *observed
                } else {
                    let observed = self.submission_timeline_counter_value(point.semaphore)?;
                    observed_timelines.insert(semaphore_key, observed);
                    observed
                };
                if observed < point.value {
                    complete = false;
                    break;
                }
            }
            if complete {
                completed_buffers.push(retirement);
            } else {
                still_in_flight_buffers.push_back(retirement);
            }
        }
        arena.in_flight_buffer_allocations = still_in_flight_buffers;
        arena.buffer_timeline_reaped = arena
            .buffer_timeline_reaped
            .saturating_add(completed_buffers.len() as u64);

        let mut still_in_flight_scratch =
            VecDeque::with_capacity(arena.in_flight_scratch_leases.len());
        let mut completed_scratch = Vec::new();
        while let Some(retirement) = arena.in_flight_scratch_leases.pop_front() {
            let mut complete = true;
            for point in &retirement.waits {
                let semaphore_key = point.semaphore.as_raw();
                let observed = if let Some(observed) = observed_timelines.get(&semaphore_key) {
                    *observed
                } else {
                    let observed = self.submission_timeline_counter_value(point.semaphore)?;
                    observed_timelines.insert(semaphore_key, observed);
                    observed
                };
                if observed < point.value {
                    complete = false;
                    break;
                }
            }
            if complete {
                completed_scratch.push(retirement);
            } else {
                still_in_flight_scratch.push_back(retirement);
            }
        }
        arena.in_flight_scratch_leases = still_in_flight_scratch;
        arena.scratch_timeline_reaped = arena
            .scratch_timeline_reaped
            .saturating_add(completed_scratch.len() as u64);
        drop(arena);

        let reaped = completed
            .len()
            .saturating_add(completed_buffers.len())
            .saturating_add(completed_scratch.len());
        let mut first_error = None;
        for mut retirement in completed {
            if let Some(profile) = retirement.kernel_timestamp_profile.take() {
                if let Err(err) = emit_kernel_timestamp_profile_for_inner(self, &profile) {
                    first_error.get_or_insert(err);
                }
                unsafe {
                    for chunk in profile.chunks {
                        self.device.destroy_query_pool(chunk.pool, None);
                    }
                }
            }
            for chunk in retirement.descriptor_pools.drain(..) {
                self.recycle_descriptor_pool_chunk(chunk);
            }
            for allocation in retirement.local_upload_allocations.drain(..) {
                self.recycle_detached_buffer_allocation(allocation);
            }
            if retirement.command_buffer != vk::CommandBuffer::null() {
                let mut ring = self
                    .command_buffer_ring
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                ring.reusable.push_back(retirement.command_buffer);
                ring.timeline_reaped = ring.timeline_reaped.saturating_add(1);
            }
        }
        for retirement in completed_buffers {
            self.recycle_detached_buffer_allocation(retirement.allocation);
        }
        for retirement in completed_scratch {
            self.release_scratch_lease(retirement.lease);
        }
        if let Some(err) = first_error {
            return Err(err);
        }
        Ok(reaped)
    }

    fn recycle_descriptor_pool_chunk(&self, chunk: DescriptorPoolArenaChunk) {
        let mut arena = self
            .submission_resource_arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        arena.reusable_descriptor_pools.push(chunk);
    }

    fn acquire_descriptor_pool_chunk(
        &self,
        required_descriptors: u32,
    ) -> Result<DescriptorPoolArenaChunk> {
        let recycled = {
            let mut arena = self
                .submission_resource_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
            let best = arena
                .reusable_descriptor_pools
                .iter()
                .enumerate()
                .filter(|(_, chunk)| chunk.storage_descriptor_capacity >= required_descriptors)
                .min_by_key(|(_, chunk)| chunk.storage_descriptor_capacity)
                .map(|(index, _)| index);
            best.map(|index| arena.reusable_descriptor_pools.swap_remove(index))
        };
        if let Some(mut chunk) = recycled {
            if let Err(err) = unsafe {
                self.device
                    .reset_descriptor_pool(chunk.pool, vk::DescriptorPoolResetFlags::empty())
            } {
                unsafe { self.device.destroy_descriptor_pool(chunk.pool, None) };
                return Err(anyhow!(
                    "resetting recycled Vulkan descriptor arena pool: {err:?}"
                ));
            }
            chunk.remaining_sets = DESCRIPTOR_ARENA_SETS_PER_POOL;
            chunk.remaining_storage_descriptors = chunk.storage_descriptor_capacity;
            let mut arena = self
                .submission_resource_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
            arena.descriptor_pool_reused = arena.descriptor_pool_reused.saturating_add(1);
            return Ok(chunk);
        }

        let descriptor_capacity =
            DESCRIPTOR_ARENA_STORAGE_DESCRIPTORS_PER_POOL.max(required_descriptors);
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: descriptor_capacity,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(DESCRIPTOR_ARENA_SETS_PER_POOL)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|err| anyhow!("creating Vulkan descriptor arena pool: {err:?}"))?;
        let mut arena = self
            .submission_resource_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
        arena.descriptor_pool_allocated = arena.descriptor_pool_allocated.saturating_add(1);
        Ok(DescriptorPoolArenaChunk {
            pool,
            remaining_sets: DESCRIPTOR_ARENA_SETS_PER_POOL,
            remaining_storage_descriptors: descriptor_capacity,
            storage_descriptor_capacity: descriptor_capacity,
        })
    }

    fn recycle_detached_buffer_allocation(&self, allocation: DetachedGpuBufferAllocation) {
        let mut arena = self
            .submission_resource_arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        arena.reusable_buffer_allocations.push(allocation);
    }

    fn register_recyclable_buffer_timeline_use(
        &self,
        buffer: vk::Buffer,
        semaphore: vk::Semaphore,
        value: u64,
    ) {
        let mut uses = self
            .recyclable_buffer_timeline_uses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let points = uses.entry(buffer.as_raw()).or_default();
        if let Some(point) = points.iter_mut().find(|point| point.semaphore == semaphore) {
            point.value = point.value.max(value);
        } else {
            points.push(SubmissionTimelinePoint { semaphore, value });
        }
    }

    fn take_recyclable_buffer_timeline_uses(
        &self,
        buffer: vk::Buffer,
    ) -> Vec<SubmissionTimelinePoint> {
        self.recyclable_buffer_timeline_uses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&buffer.as_raw())
            .unwrap_or_default()
    }

    fn retire_recyclable_buffer_after_timelines(
        &self,
        allocation: DetachedGpuBufferAllocation,
        waits: Vec<SubmissionTimelinePoint>,
    ) {
        if waits.is_empty() {
            self.recycle_detached_buffer_allocation(allocation);
            return;
        }
        let mut arena = self
            .submission_resource_arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        arena
            .in_flight_buffer_allocations
            .push_back(TimelineBufferRetirement { allocation, waits });
    }

    fn acquire_recycled_buffer_allocation(
        &self,
        required_size_bytes: usize,
        preferred_flags: vk::MemoryPropertyFlags,
        fallback_flags: vk::MemoryPropertyFlags,
    ) -> Result<Option<DetachedGpuBufferAllocation>> {
        let _ = self.reap_completed_submission_resources()?;
        let mut arena = self
            .submission_resource_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
        let choose = |required_flags: vk::MemoryPropertyFlags,
                      allocations: &[DetachedGpuBufferAllocation]| {
            allocations
                .iter()
                .enumerate()
                .filter(|(_, allocation)| {
                    allocation.size_bytes >= required_size_bytes
                        && allocation.memory_flags.contains(required_flags)
                })
                .min_by_key(|(_, allocation)| allocation.size_bytes)
                .map(|(index, _)| index)
        };
        let index = choose(preferred_flags, &arena.reusable_buffer_allocations)
            .or_else(|| choose(fallback_flags, &arena.reusable_buffer_allocations));
        let Some(index) = index else {
            return Ok(None);
        };
        let allocation = arena.reusable_buffer_allocations.swap_remove(index);
        arena.buffer_allocation_reused = arena.buffer_allocation_reused.saturating_add(1);
        Ok(Some(allocation))
    }

    fn allocate_scratch_buffer(
        &self,
        device: &VulkanDevice,
        size_bytes: usize,
        class: ScratchMemoryClass,
        preferred_flags: vk::MemoryPropertyFlags,
        fallback_flags: vk::MemoryPropertyFlags,
    ) -> Result<GpuBuffer> {
        if size_bytes == 0 {
            bail!("Vulkan scratch lease size must be positive");
        }
        let _ = self.reap_completed_submission_resources()?;
        let limits = unsafe {
            self.instance
                .get_physical_device_properties(device.physical_device)
                .limits
        };
        let descriptor_alignment = usize::try_from(limits.min_storage_buffer_offset_alignment)
            .context("Vulkan storage-buffer offset alignment exceeds host usize range")?;
        let alignment = descriptor_alignment.max(SCRATCH_SLAB_MIN_ALIGNMENT);
        if !alignment.is_power_of_two() {
            bail!("Vulkan scratch alignment {alignment} is not a power of two");
        }

        let lease = {
            let mut arena = self
                .scratch_buffer_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan scratch-buffer arena lock was poisoned"))?;
            if let Some(lease) = arena.allocate_existing(class, size_bytes, alignment, true) {
                lease
            } else {
                let capacity = scratch_slab_bytes().max(
                    align_up(size_bytes, 1024 * 1024)
                        .context("Vulkan scratch slab size overflow")?,
                );
                let slab =
                    create_scratch_slab(device, class, capacity, preferred_flags, fallback_flags)?;
                arena.push_slab(slab);
                arena
                    .allocate_existing(class, size_bytes, alignment, false)
                    .context("fresh Vulkan scratch slab cannot satisfy its first lease")?
            }
        };

        Ok(GpuBuffer {
            allocation: Arc::new(GpuBufferAllocation {
                inner: Arc::clone(&device.inner),
                device: device.clone(),
                buffer: lease.buffer,
                buffer_offset_bytes: lease.token.offset,
                memory: lease.memory,
                memory_offset: lease.memory_offset,
                allocation_span_bytes: lease.token.span_bytes,
                size_bytes,
                memory_flags: lease.memory_flags,
                dedicated_memory: false,
                recycle_on_drop: false,
                scratch_lease: Some(lease.token),
            }),
        })
    }

    fn register_scratch_lease_timeline_use(
        &self,
        lease_id: u64,
        semaphore: vk::Semaphore,
        value: u64,
    ) {
        let mut uses = self
            .scratch_lease_timeline_uses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let points = uses.entry(lease_id).or_default();
        if let Some(point) = points.iter_mut().find(|point| point.semaphore == semaphore) {
            point.value = point.value.max(value);
        } else {
            points.push(SubmissionTimelinePoint { semaphore, value });
        }
    }

    fn take_scratch_lease_timeline_uses(&self, lease_id: u64) -> Vec<SubmissionTimelinePoint> {
        self.scratch_lease_timeline_uses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&lease_id)
            .unwrap_or_default()
    }

    fn release_scratch_lease(&self, lease: ScratchLeaseToken) {
        self.scratch_buffer_arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .release(lease);
    }

    fn retire_scratch_lease_after_timelines(
        &self,
        lease: ScratchLeaseToken,
        waits: Vec<SubmissionTimelinePoint>,
    ) {
        if waits.is_empty() {
            self.release_scratch_lease(lease);
            return;
        }
        let mut arena = self
            .submission_resource_arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        arena
            .in_flight_scratch_leases
            .push_back(TimelineScratchLeaseRetirement { lease, waits });
    }

    fn retire_submission_resources_on_timeline(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_pools: Vec<DescriptorPoolArenaChunk>,
        local_upload_allocations: Vec<DetachedGpuBufferAllocation>,
        kernel_timestamp_profile: Option<KernelTimestampProfile>,
        semaphore: vk::Semaphore,
        value: u64,
    ) {
        // Registration happens after vkQueueSubmit succeeds. Recover a poisoned
        // bookkeeping mutex rather than risking early destruction of resources
        // referenced by the submitted command buffer.
        let mut arena = self
            .submission_resource_arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        arena.in_flight.push_back(TimelineSubmissionRetirement {
            command_buffer,
            descriptor_pools,
            local_upload_allocations,
            kernel_timestamp_profile,
            semaphore,
            value,
            submitted_at: Instant::now(),
        });
    }

    fn acquire_compute_command_buffer(&self) -> Result<vk::CommandBuffer> {
        // Timeline submissions detach their raw command buffers into this
        // device-owned retirement ring immediately after queue submission. Reap
        // completed entries before allocating so steady-state async workloads
        // recycle buffers without waiting for higher-level submission owners to
        // be joined or dropped.
        let _ = self.reap_completed_submission_resources()?;
        let recycled = {
            let mut ring = self
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            ring.reusable.pop_front()
        };
        if let Some(command_buffer) = recycled {
            let reset_result = {
                let _command_pool_guard = self
                    .command_pool_lock
                    .lock()
                    .map_err(|_| anyhow!("Vulkan command-pool lock was poisoned"))?;
                unsafe {
                    self.device
                        .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                }
            };
            if let Err(err) = reset_result {
                if let Ok(_command_pool_guard) = self.command_pool_lock.lock() {
                    unsafe {
                        self.device
                            .free_command_buffers(self.command_pool, &[command_buffer]);
                    }
                }
                return Err(anyhow!(
                    "resetting recycled Vulkan compute command buffer: {err:?}"
                ));
            }
            let mut ring = self
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            ring.reused = ring.reused.saturating_add(1);
            return Ok(command_buffer);
        }

        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = {
            let _command_pool_guard = self
                .command_pool_lock
                .lock()
                .map_err(|_| anyhow!("Vulkan command-pool lock was poisoned"))?;
            unsafe { self.device.allocate_command_buffers(&allocate) }
                .map_err(|err| anyhow!("allocating Vulkan compute batch command buffer: {err:?}"))?
                [0]
        };
        let mut ring = self
            .command_buffer_ring
            .lock()
            .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
        ring.allocated = ring.allocated.saturating_add(1);
        Ok(command_buffer)
    }

    fn recycle_compute_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        if command_buffer == vk::CommandBuffer::null() {
            return;
        }
        if let Ok(mut ring) = self.command_buffer_ring.lock() {
            ring.reusable.push_back(command_buffer);
        }
    }

    fn release_detached_buffer_allocation(&self, allocation: DetachedGpuBufferAllocation) {
        unsafe {
            self.device.destroy_buffer(allocation.buffer, None);
        }
        if allocation.dedicated_memory {
            unsafe { self.device.free_memory(allocation.memory, None) };
            return;
        }
        let mut allocator = self
            .memory_allocator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        allocator.free(
            &self.device,
            allocation.memory,
            allocation.memory_offset,
            allocation.allocation_span_bytes,
            allocation.size_bytes,
        );
    }

    /// Reserve a monotonically increasing completion value for one queue. This
    /// is called while the queue's external-synchronization lock is held, so
    /// values are assigned in the exact order in which submissions reach the
    /// Vulkan queue.
    fn reserve_submission_timeline(
        &self,
        queue_index: usize,
    ) -> Result<Option<(vk::Semaphore, u64)>> {
        if !self.submission_timeline_enabled {
            return Ok(None);
        }
        self.timeline_semaphore_ext
            .as_ref()
            .context("Vulkan submission timeline was enabled without KHR dispatch support")?;
        let timeline_slot = self
            .submission_timelines
            .get(queue_index)
            .context("Vulkan submission timeline queue index is out of range")?;
        let mut timeline = timeline_slot
            .lock()
            .map_err(|_| anyhow!("Vulkan submission timeline lock was poisoned"))?;
        let semaphore = match *timeline {
            Some(timeline) => timeline.semaphore,
            None => {
                let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(0);
                let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
                let semaphore = unsafe { self.device.create_semaphore(&create_info, None) }
                    .map_err(|err| {
                        anyhow!("creating Vulkan queue-completion timeline semaphore: {err:?}")
                    })?;
                *timeline = Some(QueueSubmissionTimeline { semaphore });
                semaphore
            }
        };
        let next_values = self
            .submission_timeline_next_values
            .get(queue_index)
            .context("Vulkan submission timeline counter index is out of range")?;
        let previous = next_values
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| anyhow!("Vulkan queue-completion timeline value overflow"))?;
        let value = previous
            .checked_add(1)
            .context("Vulkan queue-completion timeline value overflow")?;
        Ok(Some((semaphore, value)))
    }

    fn submission_timeline_counter_value(&self, semaphore: vk::Semaphore) -> Result<u64> {
        let timeline = self
            .timeline_semaphore_ext
            .as_ref()
            .context("Vulkan submission timeline KHR dispatch is unavailable")?;
        unsafe { timeline.get_semaphore_counter_value(semaphore) }
            .map_err(|err| anyhow!("querying Vulkan compute-batch timeline status: {err:?}"))
    }

    fn wait_submission_timeline(&self, semaphore: vk::Semaphore, value: u64) -> Result<()> {
        let timeline = self
            .timeline_semaphore_ext
            .as_ref()
            .context("Vulkan submission timeline KHR dispatch is unavailable")?;
        let semaphores = [semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { timeline.wait_semaphores(&wait_info, u64::MAX) }
            .map_err(|err| anyhow!("waiting for Vulkan compute-batch timeline: {err:?}"))
    }
}

#[derive(Clone)]
pub struct VulkanDevice {
    inner: Arc<DeviceInner>,
    physical_device: vk::PhysicalDevice,
    physical_device_index: usize,
    device_group_local_index: u32,
    device_mask: u32,
    queue_index: usize,
    name: Arc<str>,
}

/// Stable host-visible identity for one compute-capable Vulkan physical device.
/// `index` is the adapter's index in Vulkan's physical-device enumeration, so it
/// can be passed back to [`VulkanDevice::new_with_index`] without depending on
/// Hierarchos' default device-scoring policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VulkanPhysicalDeviceInfo {
    pub index: usize,
    pub name: String,
    pub device_type: String,
    pub compute_queue_family_index: u32,
    pub device_uuid: String,
    pub driver_uuid: String,
    pub device_group: Option<VulkanDeviceGroupInfo>,
    pub external_buffer: VulkanExternalBufferCapabilities,
    pub external_semaphore: VulkanExternalSemaphoreCapabilities,
}

impl VulkanPhysicalDeviceInfo {
    /// True when both entries belong to the same multi-physical-device Vulkan
    /// group returned by one enumeration pass. This is the strongest portable
    /// candidate for a future host-free cross-adapter reduction path.
    pub fn device_group_transport_candidate_with(&self, peer: &Self) -> bool {
        match (self.device_group, peer.device_group) {
            (Some(lhs), Some(rhs)) => {
                lhs.group_index == rhs.group_index
                    && lhs.physical_device_count > 1
                    && rhs.physical_device_count > 1
                    && lhs.subset_allocation
                    && rhs.subset_allocation
            }
            _ => false,
        }
    }

    /// Opaque FD/Win32 external memory is only cross-instance compatible when
    /// both Vulkan device and driver UUIDs match. Keep that compatibility rule
    /// explicit so a pair of unrelated GPUs is never mislabeled as a direct
    /// external-memory transport candidate merely because each can export its
    /// own handles.
    pub fn opaque_external_memory_transport_candidate_with(&self, peer: &Self) -> bool {
        self.device_uuid == peer.device_uuid
            && self.driver_uuid == peer.driver_uuid
            && self.external_buffer.platform_bidirectional_candidate()
            && peer.external_buffer.platform_bidirectional_candidate()
    }

    /// A usable opaque external-memory data plane also needs cross-instance GPU
    /// synchronization. Keep the semaphore probe coupled to the memory probe so
    /// callers never advertise a direct route that would still require host
    /// polling or an implicit CPU serialization point.
    pub fn opaque_external_transport_candidate_with(&self, peer: &Self) -> bool {
        self.opaque_external_memory_transport_candidate_with(peer)
            && self.external_semaphore.platform_bidirectional_candidate()
            && peer.external_semaphore.platform_bidirectional_candidate()
    }

    /// Describe the reduction transport Hierarchos can use now and the strongest
    /// host-free Vulkan route worth probing next for this device pair. Candidate
    /// discovery never upgrades the active backend by itself: device-group and
    /// external-memory paths require a real allocation/import/synchronization
    /// probe before they are safe to select for gradient payloads.
    pub fn gradient_transport_plan_with(&self, peer: &Self) -> VulkanGradientTransportPlan {
        let direct_candidate = if self.device_group_transport_candidate_with(peer) {
            Some(VulkanGradientTransportBackend::DeviceGroup)
        } else if self.opaque_external_transport_candidate_with(peer) {
            Some(VulkanGradientTransportBackend::OpaqueExternalMemory)
        } else {
            None
        };
        VulkanGradientTransportPlan {
            active_backend: VulkanGradientTransportBackend::HostVisibleStagedV2Pipelined,
            direct_candidate,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VulkanGradientTransportBackend {
    HostVisibleStagedV2Pipelined,
    DeviceGroup,
    OpaqueExternalMemory,
}

impl VulkanGradientTransportBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::HostVisibleStagedV2Pipelined => "host-visible-staged-v2-pipelined",
            Self::DeviceGroup => "device-group",
            Self::OpaqueExternalMemory => "opaque-external-memory",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanGradientTransportPlan {
    pub active_backend: VulkanGradientTransportBackend,
    pub direct_candidate: Option<VulkanGradientTransportBackend>,
}

/// Result of a live opaque external-memory/semaphore compatibility probe.
/// Success means a payload written by one logical device was imported by a
/// second logical device and read back only after the imported GPU semaphore
/// completed; capability bits alone are not enough to produce this result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanOpaqueExternalTransportProbe {
    pub handle_name: &'static str,
    pub payload_bytes: usize,
    pub synchronized_roundtrip: bool,
}

/// Vulkan 1.1 physical-device-group membership. A shared group is a stronger
/// signal than two independent adapters exposing external-memory handles: it
/// permits a future single-logical-device, device-mask transport backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanDeviceGroupInfo {
    pub group_index: usize,
    pub physical_device_count: usize,
    pub subset_allocation: bool,
}

/// Capability probe for the storage-buffer handle types relevant to an
/// inter-adapter gradient transport. These flags only establish candidacy;
/// Hierarchos must still perform a real export/import allocation probe before
/// selecting an external-memory backend for a device pair.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VulkanExternalBufferCapabilities {
    pub opaque_win32_extension_exposed: bool,
    pub opaque_win32_exportable: bool,
    pub opaque_win32_importable: bool,
    pub opaque_fd_extension_exposed: bool,
    pub opaque_fd_exportable: bool,
    pub opaque_fd_importable: bool,
}

impl VulkanExternalBufferCapabilities {
    pub fn platform_bidirectional_candidate(self) -> bool {
        #[cfg(target_os = "windows")]
        {
            return self.opaque_win32_extension_exposed
                && self.opaque_win32_exportable
                && self.opaque_win32_importable;
        }
        #[cfg(unix)]
        {
            return self.opaque_fd_extension_exposed
                && self.opaque_fd_exportable
                && self.opaque_fd_importable;
        }
        #[allow(unreachable_code)]
        false
    }

    pub fn platform_handle_name(self) -> Option<&'static str> {
        if !self.platform_bidirectional_candidate() {
            return None;
        }
        #[cfg(target_os = "windows")]
        {
            return Some("opaque-win32");
        }
        #[cfg(unix)]
        {
            return Some("opaque-fd");
        }
        #[allow(unreachable_code)]
        None
    }
}

/// Capability probe for the opaque external semaphore handle paired with an
/// external-memory transport. Like the buffer probe, this establishes
/// candidacy only; importing/exporting live handles is a separate backend step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VulkanExternalSemaphoreCapabilities {
    pub opaque_win32_extension_exposed: bool,
    pub opaque_win32_exportable: bool,
    pub opaque_win32_importable: bool,
    pub opaque_fd_extension_exposed: bool,
    pub opaque_fd_exportable: bool,
    pub opaque_fd_importable: bool,
}

impl VulkanExternalSemaphoreCapabilities {
    pub fn platform_bidirectional_candidate(self) -> bool {
        #[cfg(target_os = "windows")]
        {
            return self.opaque_win32_extension_exposed
                && self.opaque_win32_exportable
                && self.opaque_win32_importable;
        }
        #[cfg(unix)]
        {
            return self.opaque_fd_extension_exposed
                && self.opaque_fd_exportable
                && self.opaque_fd_importable;
        }
        #[allow(unreachable_code)]
        false
    }

    pub fn platform_handle_name(self) -> Option<&'static str> {
        if !self.platform_bidirectional_candidate() {
            return None;
        }
        #[cfg(target_os = "windows")]
        {
            return Some("opaque-win32");
        }
        #[cfg(unix)]
        {
            return Some("opaque-fd");
        }
        #[allow(unreachable_code)]
        None
    }
}

#[cfg(target_os = "windows")]
fn platform_external_memory_handle_type() -> vk::ExternalMemoryHandleTypeFlags {
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32
}

#[cfg(unix)]
fn platform_external_memory_handle_type() -> vk::ExternalMemoryHandleTypeFlags {
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
}

#[cfg(target_os = "windows")]
fn platform_external_semaphore_handle_type() -> vk::ExternalSemaphoreHandleTypeFlags {
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32
}

#[cfg(unix)]
fn platform_external_semaphore_handle_type() -> vk::ExternalSemaphoreHandleTypeFlags {
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD
}

#[cfg(target_os = "windows")]
fn platform_external_transport_handle_name() -> &'static str {
    "opaque-win32"
}

#[cfg(unix)]
fn platform_external_transport_handle_name() -> &'static str {
    "opaque-fd"
}

#[cfg(target_os = "windows")]
struct OwnedWin32Handle(vk::HANDLE);

#[cfg(target_os = "windows")]
impl OwnedWin32Handle {
    fn new(handle: vk::HANDLE) -> Result<Self> {
        if handle == 0 {
            bail!("Vulkan returned a null Win32 external handle");
        }
        Ok(Self(handle))
    }

    fn raw(&self) -> vk::HANDLE {
        self.0
    }
}

#[cfg(target_os = "windows")]
impl Drop for OwnedWin32Handle {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe {
                let _ = CloseHandle(self.0 as *mut core::ffi::c_void);
            }
            self.0 = 0;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanSubgroupCapabilities {
    pub subgroup_size: u32,
    pub compute_supported: bool,
    pub basic_supported: bool,
    pub arithmetic_supported: bool,
    pub shuffle_supported: bool,
    pub clustered_supported: bool,
}

/// Optional Vulkan features that can back the mixed-precision training path.
///
/// FP16 fields describe features that Hierarchos enabled on the logical
/// device. BF16 is intentionally exposure-only for now: the repository's
/// current `ash` bindings target Vulkan 1.3.281 and predate
/// `VK_KHR_shader_bfloat16`, so reporting the extension is useful for hardware
/// profiling without pretending the BF16 type feature has been queried or
/// enabled. Existing shaders, reductions, parameters, and optimizer slots stay
/// FP32 until a mixed-precision kernel explicitly opts into these capabilities.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VulkanMixedPrecisionCapabilities {
    pub storage_buffer_16_bit_access_enabled: bool,
    pub shader_float16_enabled: bool,
    pub shader_bfloat16_extension_exposed: bool,
}

impl VulkanMixedPrecisionCapabilities {
    pub fn native_fp16_storage_compute_ready(self) -> bool {
        self.storage_buffer_16_bit_access_enabled && self.shader_float16_enabled
    }
}

impl VulkanDevice {
    pub fn new() -> Result<Self> {
        Self::new_selected(None)
    }

    /// Create a logical device on an explicitly addressed Vulkan physical
    /// device. This is the addressing primitive used by multi-device training:
    /// replicas can bind to deterministic adapters instead of all selecting the
    /// same highest-scored GPU through [`VulkanDevice::new`].
    pub fn new_with_index(physical_device_index: usize) -> Result<Self> {
        Self::new_selected(Some(physical_device_index))
    }

    /// Create one Vulkan 1.1 logical device spanning the physical-device group
    /// that contains every requested adapter, then return one execution view per
    /// requested adapter. Each view records a single-bit device mask while
    /// sharing the VkDevice, queue, command pool, layout cache, and allocator.
    ///
    /// This is intentionally separate from `new_with_index`: ordinary callers
    /// retain the mature one-physical-device path, while data-parallel training
    /// can opt into device-group semantics only after discovery proved that all
    /// requested adapters are members of the same Vulkan physical-device group.
    pub fn new_device_group_with_indices(physical_device_indices: &[usize]) -> Result<Vec<Self>> {
        if physical_device_indices.len() < 2 {
            bail!("Vulkan device-group creation requires at least two physical-device indices");
        }
        if physical_device_indices.len() > 32 {
            bail!("Vulkan device groups support at most 32 physical devices");
        }
        let unique = physical_device_indices
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        if unique.len() != physical_device_indices.len() {
            bail!("Vulkan device-group indices must be unique");
        }

        let entry = unsafe { Entry::load() }.context("loading the Vulkan loader")?;
        let app_name = CString::new("Hierarchos Vulkan Trainer")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&app_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|err| anyhow!("creating Vulkan instance: {err:?}"))?;

        let create_result = (|| -> Result<(Arc<DeviceInner>, Vec<(usize, vk::PhysicalDevice, u32, Arc<str>)>)> {
            let devices = unsafe { instance.enumerate_physical_devices() }
                .map_err(|err| anyhow!("enumerating Vulkan devices for device-group creation: {err:?}"))?;
            let requested = physical_device_indices
                .iter()
                .map(|&index| {
                    devices
                        .get(index)
                        .copied()
                        .with_context(|| format!("Vulkan physical-device index {index} is out of range ({} devices)", devices.len()))
                        .map(|physical_device| (index, physical_device))
                })
                .collect::<Result<Vec<_>>>()?;

            let group_count = unsafe { instance.enumerate_physical_device_groups_len() }
                .map_err(|err| anyhow!("querying Vulkan physical-device-group count: {err:?}"))?;
            let mut groups = vec![vk::PhysicalDeviceGroupProperties::default(); group_count];
            unsafe { instance.enumerate_physical_device_groups(&mut groups) }
                .map_err(|err| anyhow!("enumerating Vulkan physical-device groups: {err:?}"))?;
            let group = groups
                .iter()
                .find(|group| {
                    let members = group.physical_devices_as_slice();
                    members.len() > 1
                        && requested
                            .iter()
                            .all(|(_, physical_device)| members.contains(physical_device))
                })
                .context("requested Vulkan adapters are not members of one multi-device physical-device group")?;
            if group.subset_allocation == vk::FALSE {
                bail!(
                    "selected Vulkan physical-device group does not support subsetAllocation; enabling one replica graph per device would replicate every model allocation across the entire group"
                );
            }
            let group_devices = group.physical_devices_as_slice().to_vec();
            if group_devices.len() > 32 {
                bail!("Vulkan physical-device group reports {} members; Hierarchos supports at most 32", group_devices.len());
            }
            let group_mask = if group_devices.len() == 32 {
                u32::MAX
            } else {
                (1u32 << group_devices.len()) - 1
            };

            let primary_physical_device = group_devices[0];
            let queue_families = unsafe {
                instance.get_physical_device_queue_family_properties(primary_physical_device)
            };
            let queue_family_index = queue_families
                .iter()
                .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .context("Vulkan device group exposes no compute queue family")? as u32;
            let mut common_queue_count = queue_families[queue_family_index as usize].queue_count;
            for &physical_device in &group_devices[1..] {
                let families = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                };
                let family = families.get(queue_family_index as usize).with_context(|| {
                    format!("Vulkan device-group member lacks queue family {queue_family_index}")
                })?;
                if !family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    bail!("Vulkan device-group queue family {queue_family_index} is not compute-capable on every member");
                }
                common_queue_count = common_queue_count.min(family.queue_count);
            }
            let queue_lane_count = usize::try_from(common_queue_count)
                .context("Vulkan device-group queue count exceeds host usize range")?
                .min(requested.len())
                .max(1);

            let mut timeline_semaphore_enabled = true;
            for &physical_device in &group_devices {
                let extensions = unsafe {
                    instance.enumerate_device_extension_properties(physical_device)
                }
                .map_err(|err| {
                    anyhow!("enumerating Vulkan device-group member extensions: {err:?}")
                })?;
                let has_timeline_extension = extensions.iter().any(|extension| unsafe {
                    CStr::from_ptr(extension.extension_name.as_ptr())
                        == vk::KHR_TIMELINE_SEMAPHORE_NAME
                });
                if !has_timeline_extension {
                    timeline_semaphore_enabled = false;
                    break;
                }
                let mut timeline_features =
                    vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
                let mut features2 =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline_features);
                unsafe {
                    instance.get_physical_device_features2(physical_device, &mut features2)
                };
                if timeline_features.timeline_semaphore == vk::FALSE {
                    timeline_semaphore_enabled = false;
                    break;
                }
            }

            let priorities = vec![1.0f32; queue_lane_count];
            let queue_infos = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let enabled_extension_names = timeline_semaphore_enabled
                .then_some(vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr())
                .into_iter()
                .collect::<Vec<_>>();
            let mut group_create_info =
                vk::DeviceGroupDeviceCreateInfo::default().physical_devices(&group_devices);
            let mut timeline_enable = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
                .timeline_semaphore(timeline_semaphore_enabled);
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&enabled_extension_names);
            if timeline_semaphore_enabled {
                device_info = device_info.push_next(&mut timeline_enable);
            }
            device_info = device_info.push_next(&mut group_create_info);
            let device = unsafe { instance.create_device(primary_physical_device, &device_info, None) }
                .map_err(|err| anyhow!("creating multi-physical-device Vulkan logical device: {err:?}"))?;
            let timeline_semaphore_ext = timeline_semaphore_enabled
                .then(|| ash::khr::timeline_semaphore::Device::new(&instance, &device));
            let queues = (0..queue_lane_count)
                .map(|queue_index| unsafe {
                    device.get_device_queue(queue_family_index, queue_index as u32)
                })
                .collect::<Vec<_>>();
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
                Ok(pool) => pool,
                Err(err) => {
                    unsafe { device.destroy_device(None) };
                    return Err(anyhow!("creating Vulkan device-group command pool: {err:?}"));
                }
            };

            let member_names = group_devices
                .iter()
                .map(|&physical_device| unsafe {
                    let properties = instance.get_physical_device_properties(physical_device);
                    CStr::from_ptr(properties.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>();
            let primary_global_index = devices
                .iter()
                .position(|candidate| *candidate == primary_physical_device)
                .context("Vulkan device-group primary disappeared from physical-device enumeration")?;
            let logical_name = format!("device-group[{}]", member_names.join(" | "));
            let inner = Arc::new(DeviceInner {
                _entry: entry,
                instance: instance.clone(),
                physical_device: primary_physical_device,
                device,
                timeline_semaphore_ext,
                queues,
                queue_locks: (0..queue_lane_count).map(|_| Mutex::new(())).collect(),
                queue_family_index,
                command_pool,
                command_pool_lock: Mutex::new(()),
                host_memory_map_lock: Mutex::new(()),
                command_buffer_ring: Mutex::new(CommandBufferRing::default()),
                submission_resource_arena: Mutex::new(TimelineSubmissionResourceArena::default()),
                recyclable_buffer_timeline_uses: Mutex::new(HashMap::new()),
                scratch_lease_timeline_uses: Mutex::new(HashMap::new()),
                scratch_buffer_arena: Mutex::new(ScratchBufferArena::default()),
                submission_timelines: (0..queue_lane_count)
                    .map(|_| Mutex::new(None))
                    .collect(),
                submission_timeline_next_values: (0..queue_lane_count)
                    .map(|_| AtomicU64::new(0))
                    .collect(),
                submission_timeline_enabled: timeline_semaphore_enabled,
                scheduler_kernel_timestamp_collection_enabled: AtomicBool::new(false),
                kernel_timestamp_profile_samples: AtomicU64::new(0),
                kernel_timestamp_profile_dispatches: AtomicU64::new(0),
                kernel_timestamp_profile_gpu_ns_total: AtomicU64::new(0),
                layout_interner: Mutex::new(LayoutInterner::default()),
                memory_allocator: Mutex::new(MemorySuballocator::default()),
                physical_device_index: primary_global_index,
                name: logical_name,
                device_group_physical_device_count: group_devices.len() as u32,
                device_group_mask: group_mask,
                device_group_timeline_semaphore_enabled: timeline_semaphore_enabled,
                opaque_external_transport_enabled: false,
                required_subgroup_size: None,
                // Device-group enablement is correctness-first for this tranche.
                // Optional FP16 features remain disabled until we intersect the
                // feature/extension sets of every physical member explicitly.
                mixed_precision_capabilities: VulkanMixedPrecisionCapabilities::default(),
            });

            let views = requested
                .into_iter()
                .map(|(global_index, physical_device)| {
                    let local_index = group_devices
                        .iter()
                        .position(|candidate| *candidate == physical_device)
                        .context("requested Vulkan adapter disappeared from selected device group")? as u32;
                    let properties = unsafe { inner.instance.get_physical_device_properties(physical_device) };
                    let name: Arc<str> = Arc::from(unsafe {
                        CStr::from_ptr(properties.device_name.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    });
                    Ok((global_index, physical_device, local_index, name))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((inner, views))
        })();

        match create_result {
            Ok((inner, views)) => Ok(views
                .into_iter()
                .map(
                    |(physical_device_index, physical_device, local_index, name)| Self {
                        inner: Arc::clone(&inner),
                        physical_device,
                        physical_device_index,
                        device_group_local_index: local_index,
                        device_mask: 1u32 << local_index,
                        queue_index: (local_index as usize) % inner.queues.len(),
                        name,
                    },
                )
                .collect()),
            Err(err) => {
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    /// Enumerate every Vulkan physical device that exposes a compute queue.
    /// The returned indices are physical-enumeration indices, not a rank after
    /// Hierarchos' discrete/integrated-GPU preference sorting.
    pub fn enumerate_compute_devices() -> Result<Vec<VulkanPhysicalDeviceInfo>> {
        let entry = unsafe { Entry::load() }.context("loading the Vulkan loader")?;
        let app_name = CString::new("Hierarchos Vulkan Trainer")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&app_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|err| anyhow!("creating Vulkan instance: {err:?}"))?;
        let result = unsafe { enumerate_compute_devices(&instance) };
        unsafe { instance.destroy_instance(None) };
        result
    }

    fn new_selected(requested_physical_device_index: Option<usize>) -> Result<Self> {
        let required_subgroup_size = match std::env::var_os(
            "HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE",
        ) {
            Some(value) => {
                let value = value.to_string_lossy();
                let parsed = value.parse::<u32>().with_context(|| {
                    format!(
                        "HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE must be a positive integer, got {value:?}"
                    )
                })?;
                if parsed == 0 || !parsed.is_power_of_two() {
                    bail!(
                        "HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE must be a nonzero power of two, got {parsed}"
                    );
                }
                Some(parsed)
            }
            None => None,
        };

        let entry = unsafe { Entry::load() }.context("loading the Vulkan loader")?;
        let app_name = CString::new("Hierarchos Vulkan Trainer")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&app_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|err| anyhow!("creating Vulkan instance: {err:?}"))?;

        let selection =
            unsafe { select_physical_device(&instance, requested_physical_device_index) };
        let (physical_device_index, physical_device, queue_family_index, name) = match selection {
            Ok(value) => value,
            Err(err) => {
                unsafe { instance.destroy_instance(None) };
                return Err(err);
            }
        };

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let extension_properties =
            match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
                Ok(properties) => properties,
                Err(err) => {
                    unsafe { instance.destroy_instance(None) };
                    return Err(anyhow!("enumerating Vulkan device extensions: {err:?}"));
                }
            };
        let has_extension = |name: &CStr| {
            extension_properties.iter().any(|extension| unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr()) == name
            })
        };
        let timeline_semaphore_extension_supported = has_extension(vk::KHR_TIMELINE_SEMAPHORE_NAME);
        let shader_float16_extension_supported = has_extension(vk::KHR_SHADER_FLOAT16_INT8_NAME);
        let shader_bfloat16_extension_exposed =
            extension_properties.iter().any(|extension| unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr()).to_bytes()
                    == b"VK_KHR_shader_bfloat16"
            });
        let external_buffer_capabilities = unsafe {
            external_buffer_capabilities(&instance, physical_device, &extension_properties)
        };
        let external_semaphore_capabilities = unsafe {
            external_semaphore_capabilities(&instance, physical_device, &extension_properties)
        };
        let opaque_external_transport_enabled = external_buffer_capabilities
            .platform_bidirectional_candidate()
            && external_semaphore_capabilities.platform_bidirectional_candidate();

        let mut storage16_support = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut shader_float16_support = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut timeline_semaphore_support = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut mixed_precision_features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut storage16_support);
        if shader_float16_extension_supported {
            mixed_precision_features =
                mixed_precision_features.push_next(&mut shader_float16_support);
        }
        if timeline_semaphore_extension_supported {
            mixed_precision_features =
                mixed_precision_features.push_next(&mut timeline_semaphore_support);
        }
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut mixed_precision_features)
        };
        let timeline_semaphore_enabled = timeline_semaphore_extension_supported
            && timeline_semaphore_support.timeline_semaphore != vk::FALSE;
        let storage_buffer_16_bit_access_enabled =
            storage16_support.storage_buffer16_bit_access != vk::FALSE;
        let shader_float16_enabled = shader_float16_extension_supported
            && shader_float16_support.shader_float16 != vk::FALSE;
        let mixed_precision_capabilities = VulkanMixedPrecisionCapabilities {
            storage_buffer_16_bit_access_enabled,
            shader_float16_enabled,
            shader_bfloat16_extension_exposed,
        };

        let mut enabled_extension_names = Vec::new();
        if timeline_semaphore_enabled {
            enabled_extension_names.push(vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr());
        }
        if shader_float16_enabled {
            enabled_extension_names.push(vk::KHR_SHADER_FLOAT16_INT8_NAME.as_ptr());
        }
        if opaque_external_transport_enabled {
            #[cfg(target_os = "windows")]
            {
                enabled_extension_names.push(vk::KHR_EXTERNAL_MEMORY_WIN32_NAME.as_ptr());
                enabled_extension_names.push(vk::KHR_EXTERNAL_SEMAPHORE_WIN32_NAME.as_ptr());
            }
            #[cfg(unix)]
            {
                enabled_extension_names.push(vk::KHR_EXTERNAL_MEMORY_FD_NAME.as_ptr());
                enabled_extension_names.push(vk::KHR_EXTERNAL_SEMAPHORE_FD_NAME.as_ptr());
            }
        }
        let mut subgroup_size_control_enable =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        if let Some(required_size) = required_subgroup_size {
            let extension_supported = has_extension(vk::EXT_SUBGROUP_SIZE_CONTROL_NAME);
            if !extension_supported {
                unsafe { instance.destroy_instance(None) };
                bail!(
                    "Vulkan device {name:?} does not expose VK_EXT_subgroup_size_control required by HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE={required_size}"
                );
            }

            let mut subgroup_size_control_features =
                vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut subgroup_size_control_features);
            unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

            let mut subgroup_size_control_properties =
                vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup_size_control_properties);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };

            let feature_supported =
                subgroup_size_control_features.subgroup_size_control != vk::FALSE;
            let compute_stage_supported = subgroup_size_control_properties
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE);
            let size_supported = required_size
                >= subgroup_size_control_properties.min_subgroup_size
                && required_size <= subgroup_size_control_properties.max_subgroup_size;
            if !feature_supported || !compute_stage_supported || !size_supported {
                unsafe { instance.destroy_instance(None) };
                bail!(
                    "Vulkan device {name:?} cannot force compute subgroup size {required_size}: feature={} compute_stage={} supported_range={}..={}",
                    feature_supported,
                    compute_stage_supported,
                    subgroup_size_control_properties.min_subgroup_size,
                    subgroup_size_control_properties.max_subgroup_size
                );
            }

            enabled_extension_names.push(vk::EXT_SUBGROUP_SIZE_CONTROL_NAME.as_ptr());
            subgroup_size_control_enable = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
                .subgroup_size_control(true);
        }
        let mut storage16_enable = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(storage_buffer_16_bit_access_enabled);
        let mut shader_float16_enable = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(shader_float16_enabled);
        let mut timeline_semaphore_enable = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
            .timeline_semaphore(timeline_semaphore_enabled);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extension_names);
        if timeline_semaphore_enabled {
            device_info = device_info.push_next(&mut timeline_semaphore_enable);
        }
        if storage_buffer_16_bit_access_enabled {
            device_info = device_info.push_next(&mut storage16_enable);
        }
        if shader_float16_enabled {
            device_info = device_info.push_next(&mut shader_float16_enable);
        }
        if required_subgroup_size.is_some() {
            device_info = device_info.push_next(&mut subgroup_size_control_enable);
        }
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(device) => device,
            Err(err) => {
                unsafe { instance.destroy_instance(None) };
                return Err(anyhow!("creating Vulkan logical device: {err:?}"));
            }
        };
        let timeline_semaphore_ext = timeline_semaphore_enabled
            .then(|| ash::khr::timeline_semaphore::Device::new(&instance, &device));
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(err) => {
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(anyhow!("creating Vulkan command pool: {err:?}"));
            }
        };

        let name: Arc<str> = Arc::from(name);
        let inner = Arc::new(DeviceInner {
            _entry: entry,
            instance,
            physical_device,
            device,
            timeline_semaphore_ext,
            queues: vec![queue],
            queue_locks: vec![Mutex::new(())],
            queue_family_index,
            command_pool,
            command_pool_lock: Mutex::new(()),
            host_memory_map_lock: Mutex::new(()),
            command_buffer_ring: Mutex::new(CommandBufferRing::default()),
            submission_resource_arena: Mutex::new(TimelineSubmissionResourceArena::default()),
            recyclable_buffer_timeline_uses: Mutex::new(HashMap::new()),
            scratch_lease_timeline_uses: Mutex::new(HashMap::new()),
            scratch_buffer_arena: Mutex::new(ScratchBufferArena::default()),
            submission_timelines: vec![Mutex::new(None)],
            submission_timeline_next_values: vec![AtomicU64::new(0)],
            submission_timeline_enabled: timeline_semaphore_enabled,
            scheduler_kernel_timestamp_collection_enabled: AtomicBool::new(false),
            kernel_timestamp_profile_samples: AtomicU64::new(0),
            kernel_timestamp_profile_dispatches: AtomicU64::new(0),
            kernel_timestamp_profile_gpu_ns_total: AtomicU64::new(0),
            layout_interner: Mutex::new(LayoutInterner::default()),
            memory_allocator: Mutex::new(MemorySuballocator::default()),
            physical_device_index,
            name: name.to_string(),
            device_group_physical_device_count: 1,
            device_group_mask: 1,
            device_group_timeline_semaphore_enabled: false,
            opaque_external_transport_enabled,
            required_subgroup_size,
            mixed_precision_capabilities,
        });
        Ok(Self {
            inner,
            physical_device,
            physical_device_index,
            device_group_local_index: 0,
            device_mask: 1,
            queue_index: 0,
            name,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the Vulkan physical-device / driver identity used to scope
    /// persistent runtime-performance evidence. Device names are not unique
    /// enough for learned scheduler state: the same marketing name may resolve
    /// to different physical adapters or materially different driver/compiler
    /// stacks across hosts.
    pub(crate) fn identity_uuids(&self) -> (String, String) {
        unsafe { physical_device_uuids(&self.inner.instance, self.physical_device) }
    }

    pub fn physical_device_index(&self) -> usize {
        self.physical_device_index
    }

    pub fn device_mask(&self) -> u32 {
        self.device_mask
    }

    pub fn device_group_mask(&self) -> u32 {
        self.inner.device_group_mask
    }

    pub fn device_group_physical_device_count(&self) -> u32 {
        self.inner.device_group_physical_device_count
    }

    pub fn is_multi_physical_device_logical_device(&self) -> bool {
        self.inner.device_group_physical_device_count > 1
    }

    pub fn shares_logical_device_with(&self, peer: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &peer.inner)
    }

    /// Enable timestamp-query collection for bounded runtime-scheduler scoring
    /// windows without requiring the process-wide diagnostic profiling
    /// environment variable. Unsupported timestamp queues fail closed to wall
    /// and timeline telemetry rather than making training fail.
    pub fn set_scheduler_kernel_timestamp_collection_enabled(&self, enabled: bool) {
        self.inner
            .scheduler_kernel_timestamp_collection_enabled
            .store(enabled, Ordering::Release);
    }

    /// Return whether bounded scheduler timestamp collection is already owned
    /// by this logical device. Runtime autotuners use this to avoid disabling a
    /// timestamp window that was opened by an outer scheduler layer.
    pub fn scheduler_kernel_timestamp_collection_enabled(&self) -> bool {
        self.inner
            .scheduler_kernel_timestamp_collection_enabled
            .load(Ordering::Acquire)
    }

    /// Time one isolated compute batch using queue timestamp queries when the
    /// selected compute queue supports them, falling back to host submit/wait
    /// latency otherwise. Scheduler-only timestamp collection stays quiet and
    /// restores the caller's prior collection ownership before returning.
    pub(crate) fn time_compute_batch_ms<F>(&self, record: F) -> Result<f64>
    where
        F: FnOnce(&mut ComputeBatch) -> Result<()>,
    {
        let timestamp_collection_was_enabled = self.scheduler_kernel_timestamp_collection_enabled();
        if !timestamp_collection_was_enabled {
            self.set_scheduler_kernel_timestamp_collection_enabled(true);
        }

        let result = (|| {
            let before = self.submission_arena_stats()?;
            let mut commands = ComputeBatch::new(self)?;
            record(&mut commands)?;
            let started = Instant::now();
            commands.submit()?;
            let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let after = self.submission_arena_stats()?;

            let sample_delta = after
                .kernel_timestamp_profile_samples
                .saturating_sub(before.kernel_timestamp_profile_samples);
            let gpu_ns_delta = after
                .kernel_timestamp_profile_gpu_ns_total
                .saturating_sub(before.kernel_timestamp_profile_gpu_ns_total);
            if sample_delta == 1 && gpu_ns_delta > 0 {
                Ok(gpu_ns_delta as f64 / 1_000_000.0)
            } else {
                Ok(wall_ms)
            }
        })();

        if !timestamp_collection_was_enabled {
            self.set_scheduler_kernel_timestamp_collection_enabled(false);
        }
        result
    }

    /// Clone this physical-device execution view while borrowing another view's
    /// queue lane. Device-group command buffers still execute under this view's
    /// device mask; only queue selection changes. This lets replica broadcast
    /// source reads run on a queue independent from primary AdamW, so AdamW may
    /// safely wait on a future timeline value before the worker submits the read.
    pub(crate) fn with_queue_lane_from(&self, lane_owner: &Self) -> Result<Self> {
        if !self.shares_logical_device_with(lane_owner)
            || !self.is_multi_physical_device_logical_device()
        {
            bail!("borrowing a Vulkan queue lane requires views of one multi-physical-device logical device");
        }
        if self.queue_index == lane_owner.queue_index {
            bail!("Vulkan device-group views do not expose independent queue lanes");
        }
        let mut view = self.clone();
        view.queue_index = lane_owner.queue_index;
        Ok(view)
    }

    pub(crate) fn opaque_external_transport_candidate_with(&self, peer: &Self) -> bool {
        if self.shares_logical_device_with(peer)
            || !self.inner.opaque_external_transport_enabled
            || !peer.inner.opaque_external_transport_enabled
        {
            return false;
        }
        let (device_uuid, driver_uuid) =
            unsafe { physical_device_uuids(&self.inner.instance, self.physical_device) };
        let (peer_device_uuid, peer_driver_uuid) =
            unsafe { physical_device_uuids(&peer.inner.instance, peer.physical_device) };
        device_uuid == peer_device_uuid && driver_uuid == peer_driver_uuid
    }

    pub fn queue_lane_count(&self) -> usize {
        self.inner.queues.len()
    }

    pub fn device_group_timeline_semaphore_enabled(&self) -> bool {
        self.inner.device_group_timeline_semaphore_enabled
    }

    /// Build two independent logical devices and prove that the selected pair
    /// can exchange an opaque external allocation under GPU semaphore control.
    /// This is intentionally separate from capability discovery: callers may
    /// use the result to decide whether an external-memory transport backend is
    /// safe to activate for the pair.
    pub fn probe_opaque_external_transport_indices(
        source_physical_device_index: usize,
        destination_physical_device_index: usize,
    ) -> Result<VulkanOpaqueExternalTransportProbe> {
        let source = Self::new_with_index(source_physical_device_index)?;
        let destination = Self::new_with_index(destination_physical_device_index)?;
        source.probe_opaque_external_transport_with(&destination)
    }

    /// Execute a live opaque external-memory + external-semaphore round trip
    /// between two independent Vulkan logical devices. The conservative UUID
    /// rule is rechecked here so a caller cannot bypass discovery and attempt
    /// opaque handles between unrelated physical identities.
    pub fn probe_opaque_external_transport_with(
        &self,
        peer: &Self,
    ) -> Result<VulkanOpaqueExternalTransportProbe> {
        if self.shares_logical_device_with(peer) {
            bail!(
                "opaque external transport probe requires two independent Vulkan logical devices"
            );
        }
        if !self.inner.opaque_external_transport_enabled
            || !peer.inner.opaque_external_transport_enabled
        {
            bail!("opaque external transport probe requires enabled bidirectional platform memory and semaphore extensions on both logical devices");
        }
        let (source_device_uuid, source_driver_uuid) =
            unsafe { physical_device_uuids(&self.inner.instance, self.physical_device) };
        let (peer_device_uuid, peer_driver_uuid) =
            unsafe { physical_device_uuids(&peer.inner.instance, peer.physical_device) };
        if source_device_uuid != peer_device_uuid || source_driver_uuid != peer_driver_uuid {
            bail!(
                "opaque external transport requires matching Vulkan device/driver UUIDs; source={source_device_uuid}/{source_driver_uuid} destination={peer_device_uuid}/{peer_driver_uuid}"
            );
        }

        let pattern = [
            0.125f32,
            -1.5,
            3.25,
            42.0,
            -0.0,
            8192.5,
            -17.75,
            std::f32::consts::PI,
        ];
        let source_payload = GpuBuffer::from_f32(self, &pattern)?;
        let source_external = create_exportable_opaque_external_f32_buffer(self, pattern.len())?;
        let destination_external =
            import_opaque_external_f32_buffer(peer, &source_external, pattern.len())?;
        let destination_readback = GpuBuffer::uninitialized_host_f32(peer, pattern.len())?;
        let semaphore_pair = OpaqueExternalSemaphorePair::new(self, peer)?;

        let mut source_commands = ComputeBatch::new(self)?;
        source_commands.copy_f32(&source_payload, &source_external, pattern.len())?;
        source_commands.release_buffer_to_external(&source_external, pattern.len())?;
        let source_submission = source_commands
            .submit_async_signal_raw_binary_semaphore(semaphore_pair.source_semaphore())?;

        let mut destination_commands = ComputeBatch::new(peer)?;
        destination_commands.acquire_buffer_from_external(&destination_external, pattern.len())?;
        destination_commands.copy_f32(
            &destination_external,
            &destination_readback,
            pattern.len(),
        )?;
        let destination_submission = destination_commands
            .submit_async_wait_raw_binary_semaphore(semaphore_pair.destination_semaphore())?;

        destination_submission.wait()?;
        source_submission.wait()?;
        let observed = destination_readback.read_f32(pattern.len())?;
        let bitwise_match = observed
            .iter()
            .map(|value| value.to_bits())
            .eq(pattern.iter().map(|value| value.to_bits()));
        if !bitwise_match {
            bail!(
                "opaque external transport round trip produced a mismatched payload: expected={pattern:?} observed={observed:?}"
            );
        }

        Ok(VulkanOpaqueExternalTransportProbe {
            handle_name: platform_external_transport_handle_name(),
            payload_bytes: pattern.len() * std::mem::size_of::<f32>(),
            synchronized_roundtrip: true,
        })
    }

    pub fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index
    }

    pub fn subgroup_capabilities(&self) -> VulkanSubgroupCapabilities {
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
        unsafe {
            self.inner
                .instance
                .get_physical_device_properties2(self.physical_device, &mut properties);
        }
        VulkanSubgroupCapabilities {
            // A required size is a per-pipeline execution contract, but this
            // device applies it uniformly to every Hierarchos compute kernel.
            // Report the effective size so geometry selection and profiling use
            // the same subgroup width that shaders actually execute with.
            subgroup_size: self
                .inner
                .required_subgroup_size
                .unwrap_or(subgroup.subgroup_size),
            compute_supported: subgroup
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE),
            basic_supported: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::BASIC),
            arithmetic_supported: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::ARITHMETIC),
            shuffle_supported: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::SHUFFLE),
            clustered_supported: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::CLUSTERED),
        }
    }

    pub fn mixed_precision_capabilities(&self) -> VulkanMixedPrecisionCapabilities {
        self.inner.mixed_precision_capabilities
    }

    pub(crate) fn supports_compute_subgroup_arithmetic(&self) -> bool {
        let caps = self.subgroup_capabilities();
        caps.subgroup_size > 0
            && caps.compute_supported
            && caps.basic_supported
            && caps.arithmetic_supported
    }

    pub(crate) fn supports_compute_subgroup_clustered_arithmetic(&self) -> bool {
        let caps = self.subgroup_capabilities();
        caps.subgroup_size >= 4
            && caps.compute_supported
            && caps.basic_supported
            && caps.arithmetic_supported
            && caps.clustered_supported
    }

    pub(crate) fn supports_compute_subgroup_shuffle(&self) -> bool {
        let caps = self.subgroup_capabilities();
        caps.subgroup_size >= 2
            && caps.compute_supported
            && caps.basic_supported
            && caps.shuffle_supported
    }

    pub(crate) fn supports_storage_buffer_bindings(&self, count: u32) -> bool {
        let properties = unsafe {
            self.inner
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        let limits = properties.limits;
        limits.max_per_stage_descriptor_storage_buffers >= count
            && limits.max_descriptor_set_storage_buffers >= count
    }

    pub(crate) fn supports_compute_work_group_size_x(&self, invocations: u32) -> bool {
        self.supports_compute_work_group_size([invocations, 1, 1])
    }

    pub(crate) fn supports_compute_work_group_size(&self, local_size: [u32; 3]) -> bool {
        if local_size.contains(&0) {
            return false;
        }
        let properties = unsafe {
            self.inner
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        let limits = properties.limits;
        let Some(invocations) = local_size
            .into_iter()
            .try_fold(1u32, |total, value| total.checked_mul(value))
        else {
            return false;
        };
        invocations <= limits.max_compute_work_group_invocations
            && local_size
                .into_iter()
                .zip(limits.max_compute_work_group_size)
                .all(|(requested, supported)| requested <= supported)
    }

    pub(crate) fn max_compute_shared_memory_bytes(&self) -> u32 {
        let properties = unsafe {
            self.inner
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        properties.limits.max_compute_shared_memory_size
    }

    /// Snapshot the allocator state used by Hierarchos storage buffers.
    ///
    /// `reserved_bytes` is the amount of Vulkan device memory currently held
    /// by pooled blocks, while `live_buffer_bytes` is the logical size of live
    /// storage buffers. The difference is reusable arena slack rather than a
    /// second copy of model tensors.
    pub fn memory_stats(&self) -> Result<VulkanMemoryStats> {
        let properties = unsafe {
            self.inner
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        let allocator = self
            .inner
            .memory_allocator
            .lock()
            .map_err(|_| anyhow!("Vulkan memory allocator lock was poisoned"))?;
        Ok(allocator.stats(properties.limits.max_memory_allocation_count))
    }

    /// Snapshot the device-owned transient submission scheduler. Unlike
    /// `memory_stats`, these counters describe command/descriptor/buffer reuse
    /// after queue timeline epochs retire rather than persistent model storage.
    pub fn submission_arena_stats(&self) -> Result<VulkanSubmissionArenaStats> {
        let _ = self.inner.reap_completed_submission_resources()?;
        let arena = self
            .inner
            .submission_resource_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
        let reusable_buffer_bytes =
            arena
                .reusable_buffer_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total
                        .checked_add(allocation.size_bytes)
                        .context("Vulkan reusable-buffer byte count overflow")
                })?;
        let scratch = self
            .inner
            .scratch_buffer_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan scratch-buffer arena lock was poisoned"))?;
        let (
            scratch_slab_count,
            scratch_slab_capacity_bytes,
            scratch_slab_free_bytes,
            scratch_live_leases,
            scratch_slab_allocated,
            scratch_lease_allocated,
            scratch_lease_reused,
        ) = scratch.stats();
        let ring = self
            .inner
            .command_buffer_ring
            .lock()
            .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
        Ok(VulkanSubmissionArenaStats {
            in_flight_submissions: arena.in_flight.len(),
            in_flight_buffer_allocations: arena.in_flight_buffer_allocations.len(),
            in_flight_scratch_leases: arena.in_flight_scratch_leases.len(),
            timeline_reaped_submissions: arena.timeline_reaped,
            timeline_retirement_latency_ns_total: arena.timeline_retirement_latency_ns_total,
            timeline_retirement_latency_ns_max: arena.timeline_retirement_latency_ns_max,
            timeline_retirement_latency_samples: arena.timeline_retirement_latency_samples,
            timeline_retirement_latency_ns_average: if arena.timeline_retirement_latency_samples
                == 0
            {
                0
            } else {
                arena.timeline_retirement_latency_ns_total
                    / arena.timeline_retirement_latency_samples
            },
            kernel_timestamp_profile_samples: self
                .inner
                .kernel_timestamp_profile_samples
                .load(Ordering::Acquire),
            kernel_timestamp_profile_dispatches: self
                .inner
                .kernel_timestamp_profile_dispatches
                .load(Ordering::Acquire),
            kernel_timestamp_profile_gpu_ns_total: self
                .inner
                .kernel_timestamp_profile_gpu_ns_total
                .load(Ordering::Acquire),
            timeline_reaped_buffer_allocations: arena.buffer_timeline_reaped,
            timeline_reaped_scratch_leases: arena.scratch_timeline_reaped,
            reusable_descriptor_pool_count: arena.reusable_descriptor_pools.len(),
            descriptor_pool_allocated: arena.descriptor_pool_allocated,
            descriptor_pool_reused: arena.descriptor_pool_reused,
            reusable_buffer_count: arena.reusable_buffer_allocations.len(),
            reusable_buffer_bytes,
            buffer_allocation_reused: arena.buffer_allocation_reused,
            scratch_slab_count,
            scratch_slab_capacity_bytes,
            scratch_slab_free_bytes,
            scratch_live_leases,
            scratch_slab_allocated,
            scratch_lease_allocated,
            scratch_lease_reused,
            reusable_command_buffer_count: ring.reusable.len(),
            command_buffer_allocated: ring.allocated,
            command_buffer_reused: ring.reused,
            command_buffer_timeline_reaped: ring.timeline_reaped,
        })
    }

    /// Query the memory working-set envelope visible to the selected Vulkan
    /// physical device.
    ///
    /// When `VK_EXT_memory_budget` is available, `device_local_budget_bytes`
    /// and `device_local_usage_bytes` are the driver's live heap budget/usage
    /// values. On older implementations the physical device-local heap sizes
    /// are used as the budget and Hierarchos' own pooled reservations are used
    /// as a conservative usage floor. The fallback is intentionally marked by
    /// `budget_extension_supported == false` so callers can distinguish an OS-
    /// aware budget from a heap-size estimate.
    pub fn memory_budget(&self) -> Result<VulkanMemoryBudget> {
        let extension_properties = unsafe {
            self.inner
                .instance
                .enumerate_device_extension_properties(self.physical_device)
        }
        .map_err(|err| {
            anyhow!("enumerating Vulkan device extensions for memory budget: {err:?}")
        })?;
        let budget_extension_supported = extension_properties.iter().any(|extension| unsafe {
            CStr::from_ptr(extension.extension_name.as_ptr()) == vk::EXT_MEMORY_BUDGET_NAME
        });

        let allocator_reserved_bytes = self
            .inner
            .memory_allocator
            .lock()
            .map_err(|_| anyhow!("Vulkan memory allocator lock was poisoned"))?
            .reserved_bytes as u64;

        let mut budget_properties = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut memory_properties2 = if budget_extension_supported {
            vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_properties)
        } else {
            vk::PhysicalDeviceMemoryProperties2::default()
        };
        unsafe {
            self.inner.instance.get_physical_device_memory_properties2(
                self.physical_device,
                &mut memory_properties2,
            );
        }
        let memory_properties = memory_properties2.memory_properties;
        let heap_count = memory_properties.memory_heap_count as usize;
        let mut heaps = Vec::with_capacity(heap_count);
        let mut device_local_heap_size_bytes = 0u64;
        let mut device_local_budget_bytes = 0u64;
        let mut device_local_driver_usage_bytes = 0u64;
        for heap_index in 0..heap_count {
            let heap = memory_properties.memory_heaps[heap_index];
            let device_local = heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
            let heap_size_bytes = heap.size;
            let budget_bytes = if budget_extension_supported {
                budget_properties.heap_budget[heap_index]
            } else {
                heap_size_bytes
            };
            let usage_bytes = if budget_extension_supported {
                budget_properties.heap_usage[heap_index]
            } else {
                0
            };
            if device_local {
                device_local_heap_size_bytes = device_local_heap_size_bytes
                    .checked_add(heap_size_bytes)
                    .context("Vulkan device-local heap size overflow")?;
                device_local_budget_bytes = device_local_budget_bytes
                    .checked_add(budget_bytes)
                    .context("Vulkan device-local budget overflow")?;
                device_local_driver_usage_bytes = device_local_driver_usage_bytes
                    .checked_add(usage_bytes)
                    .context("Vulkan device-local usage overflow")?;
            }
            heaps.push(VulkanMemoryHeapBudget {
                heap_index: heap_index as u32,
                device_local,
                heap_size_bytes,
                budget_bytes,
                usage_bytes,
            });
        }

        let device_local_usage_bytes = if budget_extension_supported {
            device_local_driver_usage_bytes
        } else {
            allocator_reserved_bytes.min(device_local_budget_bytes)
        };
        let device_local_available_bytes =
            device_local_budget_bytes.saturating_sub(device_local_usage_bytes);
        Ok(VulkanMemoryBudget {
            budget_extension_supported,
            device_local_heap_size_bytes,
            device_local_budget_bytes,
            device_local_usage_bytes,
            device_local_available_bytes,
            hierarchos_reserved_bytes: allocator_reserved_bytes,
            heaps,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VulkanMemoryStats {
    /// Number of live Vulkan storage buffers suballocated by Hierarchos.
    pub live_buffer_count: usize,
    /// Sum of requested byte sizes for live storage buffers.
    pub live_buffer_bytes: usize,
    /// Number of backing `VkDeviceMemory` allocations currently held.
    pub driver_allocation_count: usize,
    /// Total bytes reserved across the backing memory blocks.
    pub reserved_bytes: usize,
    /// Vulkan's device-wide limit for simultaneous memory allocations.
    pub max_driver_allocation_count: u32,
}

/// Device-local telemetry for Hierarchos' timeline-epoch transient scheduler.
/// These values are intentionally runtime-only and never enter checkpoint or
/// PyTorch interchange formats.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VulkanSubmissionArenaStats {
    pub in_flight_submissions: usize,
    pub in_flight_buffer_allocations: usize,
    pub in_flight_scratch_leases: usize,
    pub timeline_reaped_submissions: u64,
    /// Sum of host-observed submit-to-timeline-retirement latency for completed
    /// asynchronous command buffers. This includes device execution, queue
    /// backlog, and the bounded delay until Hierarchos next reaps the timeline.
    /// Consumers should normally difference two snapshots around a scheduler
    /// window rather than interpret this process-lifetime cumulative value.
    pub timeline_retirement_latency_ns_total: u64,
    /// Largest host-observed submit-to-retirement latency since device creation.
    pub timeline_retirement_latency_ns_max: u64,
    /// Number of asynchronous submissions contributing to the latency totals.
    pub timeline_retirement_latency_samples: u64,
    /// Integer mean of the cumulative submit-to-retirement samples.
    pub timeline_retirement_latency_ns_average: u64,
    /// Number of completed timestamp-query batches accumulated for bounded
    /// scheduler profiling or explicit diagnostic profiling.
    pub kernel_timestamp_profile_samples: u64,
    /// Number of timestamp-bracketed compute dispatches accumulated across the
    /// completed profile batches.
    pub kernel_timestamp_profile_dispatches: u64,
    /// Sum of timestamp-query GPU execution durations, in nanoseconds.
    pub kernel_timestamp_profile_gpu_ns_total: u64,
    pub timeline_reaped_buffer_allocations: u64,
    pub timeline_reaped_scratch_leases: u64,
    pub reusable_descriptor_pool_count: usize,
    pub descriptor_pool_allocated: u64,
    pub descriptor_pool_reused: u64,
    pub reusable_buffer_count: usize,
    pub reusable_buffer_bytes: usize,
    pub buffer_allocation_reused: u64,
    pub scratch_slab_count: usize,
    pub scratch_slab_capacity_bytes: usize,
    pub scratch_slab_free_bytes: usize,
    pub scratch_live_leases: usize,
    pub scratch_slab_allocated: u64,
    pub scratch_lease_allocated: u64,
    pub scratch_lease_reused: u64,
    pub reusable_command_buffer_count: usize,
    pub command_buffer_allocated: u64,
    pub command_buffer_reused: u64,
    pub command_buffer_timeline_reaped: u64,
}

/// Per-heap memory telemetry used to construct the aggregate Vulkan working-set
/// budget. `budget_bytes`/`usage_bytes` are driver values when
/// `VK_EXT_memory_budget` is available; otherwise budget equals heap size and
/// usage is zero for the individual fallback heap entries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VulkanMemoryHeapBudget {
    pub heap_index: u32,
    pub device_local: bool,
    pub heap_size_bytes: u64,
    pub budget_bytes: u64,
    pub usage_bytes: u64,
}

/// Aggregate memory budget for all device-local heaps visible to Hierarchos.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VulkanMemoryBudget {
    pub budget_extension_supported: bool,
    pub device_local_heap_size_bytes: u64,
    pub device_local_budget_bytes: u64,
    pub device_local_usage_bytes: u64,
    pub device_local_available_bytes: u64,
    /// Bytes currently reserved by Hierarchos' Vulkan suballocator across all
    /// memory types. This is useful alongside driver usage to see how much of
    /// the process working set is reusable pooled slack.
    pub hierarchos_reserved_bytes: u64,
    pub heaps: Vec<VulkanMemoryHeapBudget>,
}

impl VulkanMemoryBudget {
    /// Quantize live device-local memory pressure into eight coarse scheduler
    /// contexts. Coarse bins are deliberate: they let the contextual policy
    /// distinguish an idle GPU from a crowded one without fragmenting profile
    /// evidence on tiny driver-usage fluctuations.
    pub fn device_local_pressure_bucket(&self) -> Option<u8> {
        if self.device_local_budget_bytes == 0 {
            return None;
        }
        let scaled = self
            .device_local_usage_bytes
            .saturating_mul(8)
            .checked_div(self.device_local_budget_bytes)
            .unwrap_or(0);
        Some(u8::try_from(scaled.min(7)).unwrap_or(7))
    }
}

unsafe fn enumerate_compute_devices(instance: &Instance) -> Result<Vec<VulkanPhysicalDeviceInfo>> {
    let devices = instance
        .enumerate_physical_devices()
        .map_err(|err| anyhow!("enumerating Vulkan devices: {err:?}"))?;
    let group_count = instance
        .enumerate_physical_device_groups_len()
        .map_err(|err| anyhow!("querying Vulkan physical-device-group count: {err:?}"))?;
    let mut groups = vec![vk::PhysicalDeviceGroupProperties::default(); group_count];
    instance
        .enumerate_physical_device_groups(&mut groups)
        .map_err(|err| anyhow!("enumerating Vulkan physical-device groups: {err:?}"))?;
    let mut result = Vec::new();
    for (index, physical_device) in devices.into_iter().enumerate() {
        let properties = instance.get_physical_device_properties(physical_device);
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let Some((queue_family_index, _)) = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        let name = CStr::from_ptr(properties.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        let (device_uuid, driver_uuid) = physical_device_uuids(instance, physical_device);
        let device_group = groups.iter().enumerate().find_map(|(group_index, group)| {
            group
                .physical_devices_as_slice()
                .contains(&physical_device)
                .then_some(VulkanDeviceGroupInfo {
                    group_index,
                    physical_device_count: group.physical_device_count as usize,
                    subset_allocation: group.subset_allocation != vk::FALSE,
                })
        });
        let extension_properties = instance
            .enumerate_device_extension_properties(physical_device)
            .map_err(|err| {
                anyhow!("enumerating Vulkan device extensions for physical device {index}: {err:?}")
            })?;
        let external_buffer =
            external_buffer_capabilities(instance, physical_device, &extension_properties);
        let external_semaphore =
            external_semaphore_capabilities(instance, physical_device, &extension_properties);
        result.push(VulkanPhysicalDeviceInfo {
            index,
            name,
            device_type: format!("{:?}", properties.device_type),
            compute_queue_family_index: queue_family_index as u32,
            device_uuid,
            driver_uuid,
            device_group,
            external_buffer,
            external_semaphore,
        });
    }
    Ok(result)
}

unsafe fn physical_device_uuids(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> (String, String) {
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    instance.get_physical_device_properties2(physical_device, &mut properties);
    let format_uuid = |uuid: &[u8; vk::UUID_SIZE]| {
        uuid.iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    (format_uuid(&id.device_uuid), format_uuid(&id.driver_uuid))
}

unsafe fn external_buffer_capabilities(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    extension_properties: &[vk::ExtensionProperties],
) -> VulkanExternalBufferCapabilities {
    let has_extension = |name: &CStr| {
        extension_properties
            .iter()
            .any(|extension| CStr::from_ptr(extension.extension_name.as_ptr()) == name)
    };
    let query = |handle_type: vk::ExternalMemoryHandleTypeFlags| {
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .handle_type(handle_type);
        let mut properties = vk::ExternalBufferProperties::default();
        instance.get_physical_device_external_buffer_properties(
            physical_device,
            &info,
            &mut properties,
        );
        let features = properties
            .external_memory_properties
            .external_memory_features;
        (
            features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE),
            features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE),
        )
    };
    let (opaque_win32_exportable, opaque_win32_importable) =
        query(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let (opaque_fd_exportable, opaque_fd_importable) =
        query(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    VulkanExternalBufferCapabilities {
        opaque_win32_extension_exposed: has_extension(vk::KHR_EXTERNAL_MEMORY_WIN32_NAME),
        opaque_win32_exportable,
        opaque_win32_importable,
        opaque_fd_extension_exposed: has_extension(vk::KHR_EXTERNAL_MEMORY_FD_NAME),
        opaque_fd_exportable,
        opaque_fd_importable,
    }
}

unsafe fn external_semaphore_capabilities(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    extension_properties: &[vk::ExtensionProperties],
) -> VulkanExternalSemaphoreCapabilities {
    let has_extension = |name: &CStr| {
        extension_properties
            .iter()
            .any(|extension| CStr::from_ptr(extension.extension_name.as_ptr()) == name)
    };
    let query = |handle_type: vk::ExternalSemaphoreHandleTypeFlags| {
        let info = vk::PhysicalDeviceExternalSemaphoreInfo::default().handle_type(handle_type);
        let mut properties = vk::ExternalSemaphoreProperties::default();
        instance.get_physical_device_external_semaphore_properties(
            physical_device,
            &info,
            &mut properties,
        );
        (
            properties
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
            properties
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE),
        )
    };
    let (opaque_win32_exportable, opaque_win32_importable) =
        query(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
    let (opaque_fd_exportable, opaque_fd_importable) =
        query(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    VulkanExternalSemaphoreCapabilities {
        opaque_win32_extension_exposed: has_extension(vk::KHR_EXTERNAL_SEMAPHORE_WIN32_NAME),
        opaque_win32_exportable,
        opaque_win32_importable,
        opaque_fd_extension_exposed: has_extension(vk::KHR_EXTERNAL_SEMAPHORE_FD_NAME),
        opaque_fd_exportable,
        opaque_fd_importable,
    }
}

unsafe fn select_physical_device(
    instance: &Instance,
    requested_index: Option<usize>,
) -> Result<(usize, vk::PhysicalDevice, u32, String)> {
    let devices = instance
        .enumerate_physical_devices()
        .map_err(|err| anyhow!("enumerating Vulkan devices: {err:?}"))?;
    let mut candidates = Vec::new();
    for (index, physical_device) in devices.into_iter().enumerate() {
        let properties = instance.get_physical_device_properties(physical_device);
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let Some((queue_family_index, _)) = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        let score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        let name = CStr::from_ptr(properties.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        candidates.push((
            score,
            index,
            physical_device,
            queue_family_index as u32,
            name,
        ));
    }
    if let Some(requested_index) = requested_index {
        return candidates
            .into_iter()
            .find(|(_, index, _, _, _)| *index == requested_index)
            .map(|(_, index, device, family, name)| (index, device, family, name))
            .with_context(|| {
                format!(
                    "Vulkan physical device index {requested_index} is unavailable or exposes no compute queue"
                )
            });
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    candidates
        .into_iter()
        .next()
        .map(|(_, index, device, family, name)| (index, device, family, name))
        .context("no Vulkan device with a compute queue is available")
}

const DEFAULT_MEMORY_BLOCK_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_SCRATCH_SLAB_BYTES: usize = 16 * 1024 * 1024;
const SCRATCH_SLAB_MIN_ALIGNMENT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MemoryRange {
    offset: usize,
    size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScratchMemoryClass {
    DeviceLocal,
    HostVisible,
}

#[derive(Clone, Copy, Debug)]
struct ScratchLeaseToken {
    lease_id: u64,
    slab_id: u64,
    offset: usize,
    span_bytes: usize,
}

struct ScratchBufferSlab {
    id: u64,
    class: ScratchMemoryClass,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_offset: usize,
    allocation_span_bytes: usize,
    capacity_bytes: usize,
    memory_flags: vk::MemoryPropertyFlags,
    free_ranges: Vec<MemoryRange>,
    live_leases: usize,
}

#[derive(Clone, Copy)]
struct ScratchLeaseAllocation {
    token: ScratchLeaseToken,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_offset: usize,
    memory_flags: vk::MemoryPropertyFlags,
}

#[derive(Default)]
struct ScratchBufferArena {
    slabs: Vec<ScratchBufferSlab>,
    next_slab_id: u64,
    next_lease_id: u64,
    slab_allocated: u64,
    lease_allocated: u64,
    lease_reused: u64,
}

impl ScratchBufferArena {
    fn allocate_existing(
        &mut self,
        class: ScratchMemoryClass,
        size_bytes: usize,
        alignment: usize,
        reused: bool,
    ) -> Option<ScratchLeaseAllocation> {
        for index in 0..self.slabs.len() {
            if self.slabs[index].class != class {
                continue;
            }
            let offset =
                take_aligned_range(&mut self.slabs[index].free_ranges, size_bytes, alignment);
            let Some(offset) = offset else {
                continue;
            };
            self.next_lease_id = self.next_lease_id.saturating_add(1).max(1);
            self.lease_allocated = self.lease_allocated.saturating_add(1);
            if reused {
                self.lease_reused = self.lease_reused.saturating_add(1);
            }
            let slab = &mut self.slabs[index];
            slab.live_leases = slab.live_leases.saturating_add(1);
            return Some(ScratchLeaseAllocation {
                token: ScratchLeaseToken {
                    lease_id: self.next_lease_id,
                    slab_id: slab.id,
                    offset,
                    span_bytes: size_bytes,
                },
                buffer: slab.buffer,
                memory: slab.memory,
                memory_offset: slab.memory_offset,
                memory_flags: slab.memory_flags,
            });
        }
        None
    }

    fn push_slab(&mut self, mut slab: ScratchBufferSlab) {
        self.next_slab_id = self.next_slab_id.saturating_add(1).max(1);
        slab.id = self.next_slab_id;
        self.slab_allocated = self.slab_allocated.saturating_add(1);
        self.slabs.push(slab);
    }

    fn release(&mut self, lease: ScratchLeaseToken) {
        let Some(slab) = self.slabs.iter_mut().find(|slab| slab.id == lease.slab_id) else {
            return;
        };
        insert_free_range(
            &mut slab.free_ranges,
            MemoryRange {
                offset: lease.offset,
                size: lease.span_bytes,
            },
        );
        slab.live_leases = slab.live_leases.saturating_sub(1);
    }

    fn stats(&self) -> (usize, usize, usize, usize, u64, u64, u64) {
        let mut capacity_bytes = 0usize;
        let mut free_bytes = 0usize;
        let mut live_leases = 0usize;
        for slab in &self.slabs {
            capacity_bytes = capacity_bytes.saturating_add(slab.capacity_bytes);
            free_bytes = free_bytes.saturating_add(
                slab.free_ranges
                    .iter()
                    .fold(0usize, |total, range| total.saturating_add(range.size)),
            );
            live_leases = live_leases.saturating_add(slab.live_leases);
        }
        (
            self.slabs.len(),
            capacity_bytes,
            free_bytes,
            live_leases,
            self.slab_allocated,
            self.lease_allocated,
            self.lease_reused,
        )
    }
}

struct MemoryBlock {
    memory: vk::DeviceMemory,
    memory_type_index: u32,
    device_mask: u32,
    size_bytes: usize,
    free_ranges: Vec<MemoryRange>,
    live_allocations: usize,
}

#[derive(Clone, Copy)]
struct MemoryLease {
    memory: vk::DeviceMemory,
    offset: usize,
    span_bytes: usize,
}

#[derive(Default)]
struct MemorySuballocator {
    blocks: Vec<MemoryBlock>,
    live_buffer_count: usize,
    live_buffer_bytes: usize,
    reserved_bytes: usize,
}

impl MemorySuballocator {
    fn allocate(
        &mut self,
        device: &Device,
        memory_type_index: u32,
        device_mask: u32,
        use_device_mask: bool,
        requested_bytes: usize,
        span_bytes: usize,
        alignment: usize,
    ) -> Result<MemoryLease> {
        for index in 0..self.blocks.len() {
            if self.blocks[index].memory_type_index != memory_type_index
                || self.blocks[index].device_mask != device_mask
            {
                continue;
            }
            let lease = {
                let block = &mut self.blocks[index];
                take_aligned_range(&mut block.free_ranges, span_bytes, alignment).map(|offset| {
                    block.live_allocations = block.live_allocations.saturating_add(1);
                    MemoryLease {
                        memory: block.memory,
                        offset,
                        span_bytes,
                    }
                })
            };
            if let Some(lease) = lease {
                self.note_live_allocation(requested_bytes)?;
                return Ok(lease);
            }
        }

        let preferred_block_bytes = memory_block_bytes();
        let block_bytes = if span_bytes > preferred_block_bytes / 2 {
            align_up(span_bytes, 1024 * 1024).context("Vulkan memory block size overflow")?
        } else {
            preferred_block_bytes
        };
        let (memory, actual_block_bytes) = match allocate_memory_block(
            device,
            memory_type_index,
            device_mask,
            use_device_mask,
            block_bytes,
        ) {
            Ok(memory) => (memory, block_bytes),
            Err(primary_err) if block_bytes != span_bytes => {
                let memory = allocate_memory_block(
                    device,
                    memory_type_index,
                    device_mask,
                    use_device_mask,
                    span_bytes,
                )
                .map_err(|exact_err| {
                        anyhow!(
                            "allocating pooled Vulkan memory block ({block_bytes} bytes) failed: {primary_err:#}; exact {span_bytes}-byte retry failed: {exact_err:#}"
                        )
                    })?;
                (memory, span_bytes)
            }
            Err(err) => return Err(err),
        };

        let mut block = MemoryBlock {
            memory,
            memory_type_index,
            device_mask,
            size_bytes: actual_block_bytes,
            free_ranges: vec![MemoryRange {
                offset: 0,
                size: actual_block_bytes,
            }],
            live_allocations: 0,
        };
        let offset = take_aligned_range(&mut block.free_ranges, span_bytes, alignment)
            .context("fresh Vulkan memory block cannot satisfy its requested suballocation")?;
        block.live_allocations = 1;
        self.reserved_bytes = self
            .reserved_bytes
            .checked_add(actual_block_bytes)
            .context("Vulkan reserved-memory byte count overflow")?;
        self.note_live_allocation(requested_bytes)?;
        self.blocks.push(block);
        Ok(MemoryLease {
            memory,
            offset,
            span_bytes,
        })
    }

    fn free(
        &mut self,
        device: &Device,
        memory: vk::DeviceMemory,
        offset: usize,
        span_bytes: usize,
        requested_bytes: usize,
    ) {
        let Some(index) = self.blocks.iter().position(|block| block.memory == memory) else {
            return;
        };
        let memory_type_index = self.blocks[index].memory_type_index;
        let device_mask = self.blocks[index].device_mask;
        {
            let block = &mut self.blocks[index];
            insert_free_range(
                &mut block.free_ranges,
                MemoryRange {
                    offset,
                    size: span_bytes,
                },
            );
            block.live_allocations = block.live_allocations.saturating_sub(1);
        }
        self.live_buffer_count = self.live_buffer_count.saturating_sub(1);
        self.live_buffer_bytes = self.live_buffer_bytes.saturating_sub(requested_bytes);

        if self.blocks[index].live_allocations == 0 {
            let oversized = self.blocks[index].size_bytes > memory_block_bytes();
            let another_empty = self.blocks.iter().enumerate().any(|(other_index, block)| {
                other_index != index
                    && block.memory_type_index == memory_type_index
                    && block.device_mask == device_mask
                    && block.live_allocations == 0
            });
            if oversized || another_empty {
                let block = self.blocks.swap_remove(index);
                self.reserved_bytes = self.reserved_bytes.saturating_sub(block.size_bytes);
                unsafe { device.free_memory(block.memory, None) };
            }
        }
    }

    fn note_live_allocation(&mut self, requested_bytes: usize) -> Result<()> {
        self.live_buffer_count = self
            .live_buffer_count
            .checked_add(1)
            .context("Vulkan live-buffer count overflow")?;
        self.live_buffer_bytes = self
            .live_buffer_bytes
            .checked_add(requested_bytes)
            .context("Vulkan live-buffer byte count overflow")?;
        Ok(())
    }

    fn stats(&self, max_driver_allocation_count: u32) -> VulkanMemoryStats {
        VulkanMemoryStats {
            live_buffer_count: self.live_buffer_count,
            live_buffer_bytes: self.live_buffer_bytes,
            driver_allocation_count: self.blocks.len(),
            reserved_bytes: self.reserved_bytes,
            max_driver_allocation_count,
        }
    }

    unsafe fn release_all(&mut self, device: &Device) {
        for block in self.blocks.drain(..) {
            device.free_memory(block.memory, None);
        }
        self.live_buffer_count = 0;
        self.live_buffer_bytes = 0;
        self.reserved_bytes = 0;
    }
}

fn memory_block_bytes() -> usize {
    std::env::var("HIERARCHOS_VULKAN_MEMORY_BLOCK_MIB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&mib| (1..=1024).contains(&mib))
        .and_then(|mib| mib.checked_mul(1024 * 1024))
        .unwrap_or(DEFAULT_MEMORY_BLOCK_BYTES)
}

fn scratch_slab_bytes() -> usize {
    std::env::var("HIERARCHOS_VULKAN_SCRATCH_SLAB_MIB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&mib| (1..=1024).contains(&mib))
        .and_then(|mib| mib.checked_mul(1024 * 1024))
        .unwrap_or(DEFAULT_SCRATCH_SLAB_BYTES)
}

fn create_scratch_slab(
    device: &VulkanDevice,
    class: ScratchMemoryClass,
    capacity_bytes: usize,
    preferred: vk::MemoryPropertyFlags,
    fallback: vk::MemoryPropertyFlags,
) -> Result<ScratchBufferSlab> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(capacity_bytes as u64)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.inner.device.create_buffer(&buffer_info, None) }
        .map_err(|err| anyhow!("creating Vulkan scratch slab buffer: {err:?}"))?;
    let requirements = unsafe { device.inner.device.get_buffer_memory_requirements(buffer) };
    let (memory_type_index, memory_flags) =
        match find_memory_type_prefer(device, requirements.memory_type_bits, preferred, fallback) {
            Ok(selection) => selection,
            Err(err) => {
                unsafe { device.inner.device.destroy_buffer(buffer, None) };
                return Err(err.context("selecting Vulkan scratch slab memory type"));
            }
        };
    let span_bytes = usize::try_from(requirements.size)
        .context("Vulkan scratch slab memory requirement exceeds host usize range")?;
    let alignment = usize::try_from(requirements.alignment)
        .context("Vulkan scratch slab memory alignment exceeds host usize range")?;
    let lease = {
        let mut allocator = device
            .inner
            .memory_allocator
            .lock()
            .map_err(|_| anyhow!("Vulkan memory allocator lock was poisoned"))?;
        match allocator.allocate(
            &device.inner.device,
            memory_type_index,
            device.device_mask,
            device.is_multi_physical_device_logical_device(),
            capacity_bytes,
            span_bytes,
            alignment,
        ) {
            Ok(lease) => lease,
            Err(err) => {
                unsafe { device.inner.device.destroy_buffer(buffer, None) };
                return Err(err.context("allocating Vulkan scratch slab backing memory"));
            }
        }
    };
    if let Err(err) = unsafe {
        device
            .inner
            .device
            .bind_buffer_memory(buffer, lease.memory, lease.offset as u64)
    } {
        if let Ok(mut allocator) = device.inner.memory_allocator.lock() {
            allocator.free(
                &device.inner.device,
                lease.memory,
                lease.offset,
                lease.span_bytes,
                capacity_bytes,
            );
        }
        unsafe { device.inner.device.destroy_buffer(buffer, None) };
        return Err(anyhow!("binding Vulkan scratch slab memory: {err:?}"));
    }
    Ok(ScratchBufferSlab {
        id: 0,
        class,
        buffer,
        memory: lease.memory,
        memory_offset: lease.offset,
        allocation_span_bytes: lease.span_bytes,
        capacity_bytes,
        memory_flags,
        free_ranges: vec![MemoryRange {
            offset: 0,
            size: capacity_bytes,
        }],
        live_leases: 0,
    })
}

fn allocate_memory_block(
    device: &Device,
    memory_type_index: u32,
    device_mask: u32,
    use_device_mask: bool,
    size_bytes: usize,
) -> Result<vk::DeviceMemory> {
    let mut allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(size_bytes as u64)
        .memory_type_index(memory_type_index);
    let mut allocation_flags = vk::MemoryAllocateFlagsInfo::default()
        .flags(vk::MemoryAllocateFlags::DEVICE_MASK)
        .device_mask(device_mask);
    if use_device_mask {
        allocation = allocation.push_next(&mut allocation_flags);
    }
    unsafe { device.allocate_memory(&allocation, None) }.map_err(|err| {
        anyhow!("allocating {size_bytes} bytes of Vulkan memory type {memory_type_index} with device mask 0x{device_mask:08x}: {err:?}")
    })
}

fn take_aligned_range(
    free_ranges: &mut Vec<MemoryRange>,
    size: usize,
    alignment: usize,
) -> Option<usize> {
    if size == 0 || alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }
    for index in 0..free_ranges.len() {
        let range = free_ranges[index];
        let aligned = align_up(range.offset, alignment)?;
        let padding = aligned.checked_sub(range.offset)?;
        let consumed = padding.checked_add(size)?;
        if consumed > range.size {
            continue;
        }
        let tail_offset = aligned.checked_add(size)?;
        let tail_size = range.size - consumed;
        free_ranges.remove(index);
        if tail_size != 0 {
            free_ranges.insert(
                index,
                MemoryRange {
                    offset: tail_offset,
                    size: tail_size,
                },
            );
        }
        if padding != 0 {
            free_ranges.insert(
                index,
                MemoryRange {
                    offset: range.offset,
                    size: padding,
                },
            );
        }
        return Some(aligned);
    }
    None
}

fn insert_free_range(free_ranges: &mut Vec<MemoryRange>, range: MemoryRange) {
    if range.size == 0 {
        return;
    }
    free_ranges.push(range);
    free_ranges.sort_unstable_by_key(|range| range.offset);
    let mut merged: Vec<MemoryRange> = Vec::with_capacity(free_ranges.len());
    for range in free_ranges.drain(..) {
        if let Some(last) = merged.last_mut() {
            if let Some(last_end) = last.offset.checked_add(last.size) {
                if range.offset <= last_end {
                    let range_end = range.offset.saturating_add(range.size);
                    last.size = last.size.max(range_end.saturating_sub(last.offset));
                    continue;
                }
            }
        }
        merged.push(range);
    }
    *free_ranges = merged;
}

struct GpuBufferAllocation {
    inner: Arc<DeviceInner>,
    device: VulkanDevice,
    buffer: vk::Buffer,
    buffer_offset_bytes: usize,
    memory: vk::DeviceMemory,
    memory_offset: usize,
    allocation_span_bytes: usize,
    size_bytes: usize,
    memory_flags: vk::MemoryPropertyFlags,
    dedicated_memory: bool,
    recycle_on_drop: bool,
    scratch_lease: Option<ScratchLeaseToken>,
}

/// Reference-counted Vulkan storage buffer.
///
/// Cloning a `GpuBuffer` creates another safe view of the same Vulkan buffer
/// and device-memory allocation. The allocation is destroyed only after the
/// final view is dropped. This is the ownership primitive needed for tied
/// parameters such as `lm_head.weight`, which are consumed by multiple graph
/// branches but must retain one physical parameter identity.
#[derive(Clone)]
pub struct GpuBuffer {
    allocation: Arc<GpuBufferAllocation>,
}

impl GpuBuffer {
    fn into_detached_allocation(self) -> DetachedGpuBufferAllocation {
        let allocation = Arc::try_unwrap(self.allocation).unwrap_or_else(|_| {
            panic!("local Vulkan upload chunk unexpectedly has shared ownership")
        });
        let allocation = std::mem::ManuallyDrop::new(allocation);
        let detached = DetachedGpuBufferAllocation {
            buffer: allocation.buffer,
            memory: allocation.memory,
            memory_offset: allocation.memory_offset,
            allocation_span_bytes: allocation.allocation_span_bytes,
            size_bytes: allocation.size_bytes,
            memory_flags: allocation.memory_flags,
            dedicated_memory: allocation.dedicated_memory,
        };
        let device = unsafe { std::ptr::read(&allocation.device) };
        let inner = unsafe { std::ptr::read(&allocation.inner) };
        drop(device);
        drop(inner);
        detached
    }
}

fn create_exportable_opaque_external_f32_buffer(
    device: &VulkanDevice,
    len: usize,
) -> Result<GpuBuffer> {
    let size_bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .context("opaque external buffer size overflow")?;
    if size_bytes == 0 {
        bail!("opaque external buffer size must be positive");
    }
    if !device.inner.opaque_external_transport_enabled {
        bail!(
            "opaque external buffer allocation requires enabled platform external-memory support"
        );
    }

    let mut external_info = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(platform_external_memory_handle_type());
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size_bytes as u64)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_info);
    let buffer = unsafe { device.inner.device.create_buffer(&buffer_info, None) }
        .map_err(|err| anyhow!("creating exportable Vulkan storage buffer: {err:?}"))?;
    let requirements = unsafe { device.inner.device.get_buffer_memory_requirements(buffer) };
    let (memory_type_index, memory_flags) = match find_memory_type_prefer(
        device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    ) {
        Ok(value) => value,
        Err(err) => {
            unsafe { device.inner.device.destroy_buffer(buffer, None) };
            return Err(err.context("selecting exportable Vulkan memory type"));
        }
    };

    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(platform_external_memory_handle_type());
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
    let allocation_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index)
        .push_next(&mut export_info)
        .push_next(&mut dedicated_info);
    let memory = match unsafe { device.inner.device.allocate_memory(&allocation_info, None) } {
        Ok(memory) => memory,
        Err(err) => {
            unsafe { device.inner.device.destroy_buffer(buffer, None) };
            return Err(anyhow!("allocating exportable Vulkan memory: {err:?}"));
        }
    };
    if let Err(err) = unsafe { device.inner.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.inner.device.free_memory(memory, None);
            device.inner.device.destroy_buffer(buffer, None);
        }
        return Err(anyhow!("binding exportable Vulkan memory: {err:?}"));
    }

    Ok(GpuBuffer {
        allocation: Arc::new(GpuBufferAllocation {
            inner: Arc::clone(&device.inner),
            device: device.clone(),
            buffer,
            buffer_offset_bytes: 0,
            memory,
            memory_offset: 0,
            allocation_span_bytes: usize::try_from(requirements.size)
                .context("exportable Vulkan allocation exceeds host usize range")?,
            size_bytes,
            memory_flags,
            dedicated_memory: true,
            recycle_on_drop: false,
            scratch_lease: None,
        }),
    })
}

#[cfg(target_os = "windows")]
fn export_opaque_memory_handle(source: &GpuBuffer) -> Result<OwnedWin32Handle> {
    if !source.allocation.dedicated_memory || source.allocation.memory_offset != 0 {
        bail!("opaque external-memory export requires a dedicated zero-offset allocation");
    }
    let external_memory = ash::khr::external_memory_win32::Device::new(
        &source.allocation.inner.instance,
        &source.allocation.inner.device,
    );
    let info = vk::MemoryGetWin32HandleInfoKHR::default()
        .memory(source.allocation.memory)
        .handle_type(platform_external_memory_handle_type());
    let handle = unsafe { external_memory.get_memory_win32_handle(&info) }
        .map_err(|err| anyhow!("exporting Vulkan opaque Win32 memory handle: {err:?}"))?;
    OwnedWin32Handle::new(handle)
}

#[cfg(unix)]
fn export_opaque_memory_handle(source: &GpuBuffer) -> Result<OwnedFd> {
    if !source.allocation.dedicated_memory || source.allocation.memory_offset != 0 {
        bail!("opaque external-memory export requires a dedicated zero-offset allocation");
    }
    let external_memory = ash::khr::external_memory_fd::Device::new(
        &source.allocation.inner.instance,
        &source.allocation.inner.device,
    );
    let info = vk::MemoryGetFdInfoKHR::default()
        .memory(source.allocation.memory)
        .handle_type(platform_external_memory_handle_type());
    let fd = unsafe { external_memory.get_memory_fd(&info) }
        .map_err(|err| anyhow!("exporting Vulkan opaque memory fd: {err:?}"))?;
    if fd < 0 {
        bail!("Vulkan returned an invalid opaque memory fd {fd}");
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn import_opaque_external_f32_buffer(
    destination: &VulkanDevice,
    source: &GpuBuffer,
    len: usize,
) -> Result<GpuBuffer> {
    let size_bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .context("imported opaque external buffer size overflow")?;
    if size_bytes == 0 || size_bytes > source.allocation.size_bytes {
        bail!(
            "imported opaque external buffer size {size_bytes} is invalid for source capacity {}",
            source.allocation.size_bytes
        );
    }
    if !destination.inner.opaque_external_transport_enabled {
        bail!("opaque external-memory import requires enabled platform external-memory support");
    }

    let mut external_info = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(platform_external_memory_handle_type());
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size_bytes as u64)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_info);
    let buffer = unsafe { destination.inner.device.create_buffer(&buffer_info, None) }
        .map_err(|err| anyhow!("creating imported Vulkan storage buffer: {err:?}"))?;
    let requirements = unsafe {
        destination
            .inner
            .device
            .get_buffer_memory_requirements(buffer)
    };
    if requirements.size > source.allocation.allocation_span_bytes as u64 {
        unsafe { destination.inner.device.destroy_buffer(buffer, None) };
        bail!(
            "destination external-buffer allocation requirement {} exceeds exported source allocation {}",
            requirements.size,
            source.allocation.allocation_span_bytes
        );
    }

    #[cfg(target_os = "windows")]
    let allocation_result: Result<(vk::DeviceMemory, u32, vk::MemoryPropertyFlags)> = (|| {
        let handle = export_opaque_memory_handle(source)?;
        let external_memory = ash::khr::external_memory_win32::Device::new(
            &destination.inner.instance,
            &destination.inner.device,
        );
        let mut handle_properties = vk::MemoryWin32HandlePropertiesKHR::default();
        unsafe {
            external_memory.get_memory_win32_handle_properties(
                platform_external_memory_handle_type(),
                handle.raw(),
                &mut handle_properties,
            )
        }
        .map_err(|err| anyhow!("querying imported opaque Win32 memory properties: {err:?}"))?;
        let compatible_bits = requirements.memory_type_bits & handle_properties.memory_type_bits;
        let (memory_type_index, memory_flags) = find_memory_type_prefer(
            destination,
            compatible_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )
        .context("selecting imported opaque Win32 memory type")?;
        let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::default()
            .handle_type(platform_external_memory_handle_type())
            .handle(handle.raw());
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);
        let memory = unsafe {
            destination
                .inner
                .device
                .allocate_memory(&allocation_info, None)
        }
        .map_err(|err| anyhow!("importing opaque Win32 Vulkan memory: {err:?}"))?;
        Ok((memory, memory_type_index, memory_flags))
    })();

    #[cfg(unix)]
    let allocation_result: Result<(vk::DeviceMemory, u32, vk::MemoryPropertyFlags)> = (|| {
        let handle = export_opaque_memory_handle(source)?;
        let external_memory = ash::khr::external_memory_fd::Device::new(
            &destination.inner.instance,
            &destination.inner.device,
        );
        let mut handle_properties = vk::MemoryFdPropertiesKHR::default();
        unsafe {
            external_memory.get_memory_fd_properties(
                platform_external_memory_handle_type(),
                handle.as_raw_fd(),
                &mut handle_properties,
            )
        }
        .map_err(|err| anyhow!("querying imported opaque memory-fd properties: {err:?}"))?;
        let compatible_bits = requirements.memory_type_bits & handle_properties.memory_type_bits;
        let (memory_type_index, memory_flags) = find_memory_type_prefer(
            destination,
            compatible_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )
        .context("selecting imported opaque memory-fd type")?;
        let raw_fd = handle.into_raw_fd();
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(platform_external_memory_handle_type())
            .fd(raw_fd);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);
        match unsafe {
            destination
                .inner
                .device
                .allocate_memory(&allocation_info, None)
        } {
            Ok(memory) => Ok((memory, memory_type_index, memory_flags)),
            Err(err) => {
                drop(unsafe { OwnedFd::from_raw_fd(raw_fd) });
                Err(anyhow!("importing opaque Vulkan memory fd: {err:?}"))
            }
        }
    })();

    let (memory, _memory_type_index, memory_flags) = match allocation_result {
        Ok(value) => value,
        Err(err) => {
            unsafe { destination.inner.device.destroy_buffer(buffer, None) };
            return Err(err);
        }
    };
    if let Err(err) = unsafe {
        destination
            .inner
            .device
            .bind_buffer_memory(buffer, memory, 0)
    } {
        unsafe {
            destination.inner.device.free_memory(memory, None);
            destination.inner.device.destroy_buffer(buffer, None);
        }
        return Err(anyhow!("binding imported opaque Vulkan memory: {err:?}"));
    }

    Ok(GpuBuffer {
        allocation: Arc::new(GpuBufferAllocation {
            inner: Arc::clone(&destination.inner),
            device: destination.clone(),
            buffer,
            buffer_offset_bytes: 0,
            memory,
            memory_offset: 0,
            allocation_span_bytes: usize::try_from(requirements.size)
                .context("imported Vulkan allocation exceeds host usize range")?,
            size_bytes,
            memory_flags,
            dedicated_memory: true,
            recycle_on_drop: false,
            scratch_lease: None,
        }),
    })
}

impl GpuBuffer {
    pub(crate) fn f32_capacity(&self) -> usize {
        self.allocation.size_bytes / std::mem::size_of::<f32>()
    }

    fn buffer_region_key(&self) -> BufferRegionKey {
        BufferRegionKey {
            buffer: self.allocation.buffer.as_raw(),
            offset: self.allocation.buffer_offset_bytes as u64,
            size: self.allocation.size_bytes as u64,
        }
    }

    fn absolute_buffer_offset(&self, relative_offset: usize) -> Result<usize> {
        self.allocation
            .buffer_offset_bytes
            .checked_add(relative_offset)
            .context("Vulkan buffer-view offset overflow")
    }

    /// Allocate short-lived device scratch as an offset lease inside a
    /// persistent device-local slab `VkBuffer`. The lease is returned only
    /// after every timeline epoch that referenced it has completed.
    pub(crate) fn transient_f32(device: &VulkanDevice, len: usize) -> Result<Self> {
        let size_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .context("transient Vulkan FP32 buffer size overflow")?;
        device.inner.allocate_scratch_buffer(
            device,
            size_bytes,
            ScratchMemoryClass::DeviceLocal,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    }

    /// Allocate short-lived host-visible FP32 scratch as an offset lease inside
    /// a persistent mapped/readback-capable slab `VkBuffer`.
    pub(crate) fn transient_host_f32(device: &VulkanDevice, len: usize) -> Result<Self> {
        let size_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .context("transient host-visible Vulkan FP32 buffer size overflow")?;
        device.inner.allocate_scratch_buffer(
            device,
            size_bytes,
            ScratchMemoryClass::HostVisible,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    }

    pub fn zeros_f32(device: &VulkanDevice, len: usize) -> Result<Self> {
        let values = vec![0.0f32; len];
        Self::from_f32(device, &values)
    }

    pub fn zeros_u32(device: &VulkanDevice, len: usize) -> Result<Self> {
        let values = vec![0u32; len];
        Self::from_u32(device, &values)
    }

    pub(crate) fn zeros_host_f32(device: &VulkanDevice, len: usize) -> Result<Self> {
        let size_bytes = len.checked_mul(4).context("buffer size overflow")?;
        let buffer = Self::new_host_visible(device, size_bytes)?;
        buffer.write_mapped(bytemuck::cast_slice(&vec![0.0f32; len]))?;
        Ok(buffer)
    }

    /// Allocate a host-visible FP32 transport window without a temporary Rust
    /// heap payload. The cross-adapter streamer overwrites the active range only
    /// after the producer fence has completed.
    pub(crate) fn uninitialized_host_f32(device: &VulkanDevice, len: usize) -> Result<Self> {
        let size_bytes = len.checked_mul(4).context("buffer size overflow")?;
        Self::new_host_visible(device, size_bytes)
    }

    /// Allocate one device-local FP32 transport window on `source` and import
    /// the same opaque allocation into an independent destination VkDevice.
    /// Both returned buffers alias one physical allocation; ownership is moved
    /// between queue families with external-memory barriers and an external
    /// semaphore by the caller.
    pub(crate) fn uninitialized_opaque_external_pair_f32(
        source: &VulkanDevice,
        destination: &VulkanDevice,
        len: usize,
    ) -> Result<(Self, Self)> {
        if !source.opaque_external_transport_candidate_with(destination) {
            bail!(
                "opaque external transport requires independent logical devices with matching Vulkan device/driver UUIDs and enabled platform memory/semaphore support"
            );
        }
        let source_window = create_exportable_opaque_external_f32_buffer(source, len)?;
        let destination_window =
            import_opaque_external_f32_buffer(destination, &source_window, len)?;
        Ok((source_window, destination_window))
    }

    /// Allocate a device-group transport buffer whose resource on every
    /// physical member is bound to the destination view's memory instance.
    /// A replica can therefore DMA its gradient slice directly into memory that
    /// the primary reduction kernels consume, without mapping or copying it on
    /// the CPU. The allocation is accepted only when Vulkan reports COPY_DST
    /// peer-memory support from `source` to the destination memory heap.
    pub(crate) fn uninitialized_device_group_peer_f32(
        destination: &VulkanDevice,
        source: &VulkanDevice,
        len: usize,
    ) -> Result<Self> {
        if !destination.shares_logical_device_with(source)
            || !destination.is_multi_physical_device_logical_device()
        {
            bail!("device-group peer transport requires two views of one multi-physical-device Vulkan logical device");
        }
        if destination.device_mask == source.device_mask {
            bail!(
                "device-group peer transport requires distinct source and destination device masks"
            );
        }
        let size_bytes = len
            .checked_mul(4)
            .context("device-group buffer size overflow")?;
        if size_bytes == 0 {
            bail!("device-group transport buffer size must be positive");
        }
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { destination.inner.device.create_buffer(&buffer_info, None) }
            .map_err(|err| anyhow!("creating Vulkan device-group transport buffer: {err:?}"))?;
        let requirements = unsafe {
            destination
                .inner
                .device
                .get_buffer_memory_requirements(buffer)
        };
        let memory_properties = unsafe {
            destination
                .inner
                .instance
                .get_physical_device_memory_properties(destination.physical_device)
        };
        let select_memory_type = |required: vk::MemoryPropertyFlags| {
            (0..memory_properties.memory_type_count).find_map(|memory_type_index| {
                if requirements.memory_type_bits & (1 << memory_type_index) == 0 {
                    return None;
                }
                let memory_type = memory_properties.memory_types[memory_type_index as usize];
                if !memory_type.property_flags.contains(required) {
                    return None;
                }
                let peer_features = unsafe {
                    destination
                        .inner
                        .device
                        .get_device_group_peer_memory_features(
                            memory_type.heap_index,
                            source.device_group_local_index,
                            destination.device_group_local_index,
                        )
                };
                peer_features
                    .contains(vk::PeerMemoryFeatureFlags::COPY_DST)
                    .then_some((memory_type_index, memory_type.property_flags))
            })
        };
        let Some((memory_type_index, memory_flags)) =
            select_memory_type(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .or_else(|| select_memory_type(vk::MemoryPropertyFlags::empty()))
        else {
            unsafe { destination.inner.device.destroy_buffer(buffer, None) };
            bail!(
                "Vulkan device-group pair {} -> {} exposes no compatible peer-memory heap with COPY_DST",
                source.physical_device_index,
                destination.physical_device_index
            );
        };

        let mut allocation_flags = vk::MemoryAllocateFlagsInfo::default()
            .flags(vk::MemoryAllocateFlags::DEVICE_MASK)
            // subsetAllocation was required when the logical device was built,
            // so allocate only the destination/primary memory instance. Every
            // resource device index below is explicitly rebound to this one
            // instance; allocating the whole group here would duplicate the
            // transport window on every adapter for no benefit.
            .device_mask(destination.device_mask);
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut allocation_flags);
        let memory = match unsafe {
            destination
                .inner
                .device
                .allocate_memory(&allocation_info, None)
        } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { destination.inner.device.destroy_buffer(buffer, None) };
                return Err(anyhow!(
                    "allocating Vulkan device-group transport memory: {err:?}"
                ));
            }
        };

        // Bind every resource device index to the primary/destination memory
        // instance. This is the crucial cross-adapter mapping: source-device
        // transfer commands write the same physical allocation that primary
        // compute descriptors subsequently read.
        let device_indices = vec![
            destination.device_group_local_index;
            destination.inner.device_group_physical_device_count as usize
        ];
        let mut group_bind =
            vk::BindBufferMemoryDeviceGroupInfo::default().device_indices(&device_indices);
        let bind = vk::BindBufferMemoryInfo::default()
            .buffer(buffer)
            .memory(memory)
            .memory_offset(0)
            .push_next(&mut group_bind);
        if let Err(err) = unsafe { destination.inner.device.bind_buffer_memory2(&[bind]) } {
            unsafe {
                destination.inner.device.free_memory(memory, None);
                destination.inner.device.destroy_buffer(buffer, None);
            }
            return Err(anyhow!(
                "binding Vulkan device-group transport memory: {err:?}"
            ));
        }

        Ok(Self {
            allocation: Arc::new(GpuBufferAllocation {
                inner: Arc::clone(&destination.inner),
                device: destination.clone(),
                buffer,
                buffer_offset_bytes: 0,
                memory,
                memory_offset: 0,
                allocation_span_bytes: usize::try_from(requirements.size)
                    .context("Vulkan device-group allocation exceeds host usize range")?,
                size_bytes,
                memory_flags,
                dedicated_memory: true,
                recycle_on_drop: false,
                scratch_lease: None,
            }),
        })
    }

    pub fn from_f32(device: &VulkanDevice, values: &[f32]) -> Result<Self> {
        let buffer = Self::new(
            device,
            values
                .len()
                .checked_mul(4)
                .context("buffer size overflow")?,
        )?;
        buffer.write_f32(values)?;
        Ok(buffer)
    }

    pub fn from_u32(device: &VulkanDevice, values: &[u32]) -> Result<Self> {
        let buffer = Self::new(
            device,
            values
                .len()
                .checked_mul(4)
                .context("buffer size overflow")?,
        )?;
        buffer.write_u32(values)?;
        Ok(buffer)
    }

    fn new(device: &VulkanDevice, size_bytes: usize) -> Result<Self> {
        Self::new_with_memory(
            device,
            size_bytes,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    }

    fn new_host_visible(device: &VulkanDevice, size_bytes: usize) -> Result<Self> {
        Self::new_with_memory(
            device,
            size_bytes,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    }

    fn new_recyclable_host_visible(device: &VulkanDevice, size_bytes: usize) -> Result<Self> {
        Self::new_with_memory_policy(
            device,
            size_bytes,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            true,
        )
    }

    fn new_with_memory(
        device: &VulkanDevice,
        size_bytes: usize,
        preferred: vk::MemoryPropertyFlags,
        fallback: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        Self::new_with_memory_policy(device, size_bytes, preferred, fallback, false)
    }

    fn new_with_memory_policy(
        device: &VulkanDevice,
        size_bytes: usize,
        preferred: vk::MemoryPropertyFlags,
        fallback: vk::MemoryPropertyFlags,
        recycle_on_drop: bool,
    ) -> Result<Self> {
        if size_bytes == 0 {
            bail!("Vulkan storage buffer size must be positive");
        }
        if recycle_on_drop {
            if let Some(allocation) = device
                .inner
                .acquire_recycled_buffer_allocation(size_bytes, preferred, fallback)?
            {
                return Ok(Self {
                    allocation: Arc::new(GpuBufferAllocation {
                        inner: Arc::clone(&device.inner),
                        device: device.clone(),
                        buffer: allocation.buffer,
                        buffer_offset_bytes: 0,
                        memory: allocation.memory,
                        memory_offset: allocation.memory_offset,
                        allocation_span_bytes: allocation.allocation_span_bytes,
                        size_bytes: allocation.size_bytes,
                        memory_flags: allocation.memory_flags,
                        dedicated_memory: allocation.dedicated_memory,
                        recycle_on_drop: true,
                        scratch_lease: None,
                    }),
                });
            }
        }
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.inner.device.create_buffer(&buffer_info, None) }
            .map_err(|err| anyhow!("creating Vulkan storage buffer: {err:?}"))?;
        let requirements = unsafe { device.inner.device.get_buffer_memory_requirements(buffer) };
        let (memory_type_index, memory_flags) =
            find_memory_type_prefer(device, requirements.memory_type_bits, preferred, fallback)?;
        let span_bytes = usize::try_from(requirements.size)
            .context("Vulkan buffer memory requirement exceeds host usize range")?;
        let alignment = usize::try_from(requirements.alignment)
            .context("Vulkan buffer alignment exceeds host usize range")?;
        let mut allocator = device
            .inner
            .memory_allocator
            .lock()
            .map_err(|_| anyhow!("Vulkan memory allocator lock was poisoned"))?;
        let lease = match allocator.allocate(
            &device.inner.device,
            memory_type_index,
            device.device_mask,
            device.is_multi_physical_device_logical_device(),
            size_bytes,
            span_bytes,
            alignment,
        ) {
            Ok(lease) => lease,
            Err(err) => {
                let stats = allocator.stats(
                    unsafe {
                        device
                            .inner
                            .instance
                            .get_physical_device_properties(device.physical_device)
                    }
                    .limits
                    .max_memory_allocation_count,
                );
                unsafe { device.inner.device.destroy_buffer(buffer, None) };
                return Err(anyhow!(
                    "allocating Vulkan storage for {size_bytes} bytes: {err:#}; allocator has {} live buffers / {} bytes in {} driver allocations / {} reserved bytes (device allocation limit {}). Reduce batch/tape capacity or set HIERARCHOS_VULKAN_MEMORY_BLOCK_MIB to tune arena block size",
                    stats.live_buffer_count,
                    stats.live_buffer_bytes,
                    stats.driver_allocation_count,
                    stats.reserved_bytes,
                    stats.max_driver_allocation_count,
                ));
            }
        };
        let bind_result = {
            let _map_guard = device
                .inner
                .host_memory_map_lock
                .lock()
                .map_err(|_| anyhow!("Vulkan host-memory map lock was poisoned"))?;
            unsafe {
                device
                    .inner
                    .device
                    .bind_buffer_memory(buffer, lease.memory, lease.offset as u64)
            }
        };
        if let Err(err) = bind_result {
            allocator.free(
                &device.inner.device,
                lease.memory,
                lease.offset,
                lease.span_bytes,
                size_bytes,
            );
            unsafe { device.inner.device.destroy_buffer(buffer, None) };
            return Err(anyhow!("binding Vulkan buffer memory: {err:?}"));
        }
        drop(allocator);
        Ok(Self {
            allocation: Arc::new(GpuBufferAllocation {
                inner: Arc::clone(&device.inner),
                device: device.clone(),
                buffer,
                buffer_offset_bytes: 0,
                memory: lease.memory,
                memory_offset: lease.offset,
                allocation_span_bytes: lease.span_bytes,
                size_bytes,
                memory_flags,
                dedicated_memory: false,
                recycle_on_drop,
                scratch_lease: None,
            }),
        })
    }

    pub fn write_f32(&self, values: &[f32]) -> Result<()> {
        if values.iter().any(|value| !value.is_finite()) {
            bail!("refusing to upload non-finite FP32 values");
        }
        self.write_bytes(bytemuck::cast_slice(values))
    }

    pub fn write_u32(&self, values: &[u32]) -> Result<()> {
        self.write_bytes(bytemuck::cast_slice(values))
    }

    fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.allocation.size_bytes {
            bail!(
                "upload of {} bytes exceeds buffer size {}",
                bytes.len(),
                self.allocation.size_bytes
            );
        }
        if self
            .allocation
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            self.write_mapped(bytes)?;
        } else {
            let device = self.allocation.device.clone();
            let staging = Self::new_host_visible(&device, bytes.len())?;
            staging.write_mapped(bytes)?;
            copy_buffer(&device, &staging, self, bytes.len())?;
        }
        Ok(())
    }

    fn write_mapped(&self, bytes: &[u8]) -> Result<()> {
        self.write_mapped_range(0, bytes)
    }

    fn write_mapped_range(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(bytes.len())
            .context("mapped upload range overflow")?;
        if end > self.allocation.size_bytes {
            bail!(
                "mapped upload range {offset}..{end} exceeds buffer size {}",
                self.allocation.size_bytes
            );
        }
        let _map_guard = self
            .allocation
            .inner
            .host_memory_map_lock
            .lock()
            .map_err(|_| anyhow!("Vulkan host-memory map lock was poisoned"))?;
        let mapped = unsafe {
            self.allocation.inner.device.map_memory(
                self.allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|err| anyhow!("mapping Vulkan memory for upload: {err:?}"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped
                    .cast::<u8>()
                    .add(self.allocation.memory_offset)
                    .add(self.allocation.buffer_offset_bytes)
                    .add(offset),
                bytes.len(),
            );
            if !self
                .allocation
                .memory_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                let range = [vk::MappedMemoryRange::default()
                    .memory(self.allocation.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                self.allocation
                    .inner
                    .device
                    .flush_mapped_memory_ranges(&range)
                    .map_err(|err| anyhow!("flushing Vulkan upload memory: {err:?}"))?;
            }
            self.allocation
                .inner
                .device
                .unmap_memory(self.allocation.memory);
        }
        Ok(())
    }

    pub fn read_f32(&self, len: usize) -> Result<Vec<f32>> {
        let bytes = len.checked_mul(4).context("read size overflow")?;
        if bytes > self.allocation.size_bytes {
            bail!(
                "read of {bytes} bytes exceeds buffer size {}",
                self.allocation.size_bytes
            );
        }
        if self
            .allocation
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return self.read_mapped_f32(len);
        }
        let device = self.allocation.device.clone();
        let staging = Self::new_host_visible(&device, bytes)?;
        copy_buffer(&device, self, &staging, bytes)?;
        staging.read_mapped_f32(len)
    }

    pub fn read_u32(&self, len: usize) -> Result<Vec<u32>> {
        let bytes = len.checked_mul(4).context("read size overflow")?;
        if bytes > self.allocation.size_bytes {
            bail!(
                "read of {bytes} bytes exceeds buffer size {}",
                self.allocation.size_bytes
            );
        }
        if self
            .allocation
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return self.read_mapped_u32(len);
        }
        let device = self.allocation.device.clone();
        let staging = Self::new_host_visible(&device, bytes)?;
        copy_buffer(&device, self, &staging, bytes)?;
        staging.read_mapped_u32(len)
    }

    fn read_mapped_f32(&self, len: usize) -> Result<Vec<f32>> {
        let _map_guard = self
            .allocation
            .inner
            .host_memory_map_lock
            .lock()
            .map_err(|_| anyhow!("Vulkan host-memory map lock was poisoned"))?;
        let mapped = unsafe {
            self.allocation.inner.device.map_memory(
                self.allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|err| anyhow!("mapping Vulkan memory for readback: {err:?}"))?;
        let mut values = vec![0.0f32; len];
        unsafe {
            if !self
                .allocation
                .memory_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                let range = [vk::MappedMemoryRange::default()
                    .memory(self.allocation.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                self.allocation
                    .inner
                    .device
                    .invalidate_mapped_memory_ranges(&range)
                    .map_err(|err| anyhow!("invalidating Vulkan readback memory: {err:?}"))?;
            }
            std::ptr::copy_nonoverlapping(
                mapped
                    .cast::<u8>()
                    .add(self.allocation.memory_offset)
                    .add(self.allocation.buffer_offset_bytes)
                    .cast::<f32>(),
                values.as_mut_ptr(),
                len,
            );
            self.allocation
                .inner
                .device
                .unmap_memory(self.allocation.memory);
        }
        Ok(values)
    }

    fn read_mapped_u32(&self, len: usize) -> Result<Vec<u32>> {
        let _map_guard = self
            .allocation
            .inner
            .host_memory_map_lock
            .lock()
            .map_err(|_| anyhow!("Vulkan host-memory map lock was poisoned"))?;
        let mapped = unsafe {
            self.allocation.inner.device.map_memory(
                self.allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|err| anyhow!("mapping Vulkan memory for readback: {err:?}"))?;
        let mut values = vec![0u32; len];
        unsafe {
            if !self
                .allocation
                .memory_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                let range = [vk::MappedMemoryRange::default()
                    .memory(self.allocation.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                self.allocation
                    .inner
                    .device
                    .invalidate_mapped_memory_ranges(&range)
                    .map_err(|err| anyhow!("invalidating Vulkan readback memory: {err:?}"))?;
            }
            std::ptr::copy_nonoverlapping(
                mapped
                    .cast::<u8>()
                    .add(self.allocation.memory_offset)
                    .add(self.allocation.buffer_offset_bytes)
                    .cast::<u32>(),
                values.as_mut_ptr(),
                len,
            );
            self.allocation
                .inner
                .device
                .unmap_memory(self.allocation.memory);
        }
        Ok(values)
    }

    /// Copy a bounded FP32 payload directly between two host-visible Vulkan
    /// allocations, including allocations owned by different logical devices.
    /// This is the CPU transport seam used by cross-adapter gradient streaming:
    /// no model-sized host snapshot or per-chunk heap `Vec<f32>` is created.
    pub(crate) fn copy_host_visible_f32_to(&self, dst: &Self, len: usize) -> Result<()> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .context("host-visible Vulkan copy size overflow")?;
        if bytes == 0 || bytes > self.allocation.size_bytes || bytes > dst.allocation.size_bytes {
            bail!(
                "host-visible Vulkan copy of {bytes} bytes exceeds source/destination capacities {}/{}",
                self.allocation.size_bytes,
                dst.allocation.size_bytes
            );
        }
        if !self
            .allocation
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            || !dst
                .allocation
                .memory_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            bail!("cross-device host copy requires host-visible source and destination buffers");
        }

        if Arc::ptr_eq(&self.allocation.inner, &dst.allocation.inner) {
            let _map_guard = self
                .allocation
                .inner
                .host_memory_map_lock
                .lock()
                .map_err(|_| anyhow!("Vulkan host-memory map lock was poisoned"))?;
            return self.copy_host_visible_f32_to_locked(dst, bytes);
        }

        // Cross-adapter gradient transport can run in both directions. Acquire
        // the two device-owned map locks in a stable address order so concurrent
        // opposing copies cannot deadlock while each keeps both VkDeviceMemory
        // mappings alive for the duration of the memcpy.
        let source_key = Arc::as_ptr(&self.allocation.inner) as usize;
        let destination_key = Arc::as_ptr(&dst.allocation.inner) as usize;
        if source_key < destination_key {
            let _source_guard = self
                .allocation
                .inner
                .host_memory_map_lock
                .lock()
                .map_err(|_| anyhow!("Vulkan source host-memory map lock was poisoned"))?;
            let _destination_guard = dst
                .allocation
                .inner
                .host_memory_map_lock
                .lock()
                .map_err(|_| anyhow!("Vulkan destination host-memory map lock was poisoned"))?;
            self.copy_host_visible_f32_to_locked(dst, bytes)
        } else {
            let _destination_guard = dst
                .allocation
                .inner
                .host_memory_map_lock
                .lock()
                .map_err(|_| anyhow!("Vulkan destination host-memory map lock was poisoned"))?;
            let _source_guard = self
                .allocation
                .inner
                .host_memory_map_lock
                .lock()
                .map_err(|_| anyhow!("Vulkan source host-memory map lock was poisoned"))?;
            self.copy_host_visible_f32_to_locked(dst, bytes)
        }
    }

    fn copy_host_visible_f32_to_locked(&self, dst: &Self, bytes: usize) -> Result<()> {
        debug_assert!(bytes > 0);

        if Arc::ptr_eq(&self.allocation.inner, &dst.allocation.inner)
            && self.allocation.memory == dst.allocation.memory
        {
            let mapped = unsafe {
                self.allocation.inner.device.map_memory(
                    self.allocation.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|err| anyhow!("mapping shared Vulkan memory for host copy: {err:?}"))?;
            let copy_result = unsafe {
                if !self
                    .allocation
                    .memory_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
                {
                    let ranges = [vk::MappedMemoryRange::default()
                        .memory(self.allocation.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)];
                    self.allocation
                        .inner
                        .device
                        .invalidate_mapped_memory_ranges(&ranges)
                        .map_err(|err| anyhow!("invalidating shared host-copy memory: {err:?}"))?;
                }
                std::ptr::copy(
                    mapped
                        .cast::<u8>()
                        .add(self.allocation.memory_offset)
                        .add(self.allocation.buffer_offset_bytes),
                    mapped
                        .cast::<u8>()
                        .add(dst.allocation.memory_offset)
                        .add(dst.allocation.buffer_offset_bytes),
                    bytes,
                );
                if !dst
                    .allocation
                    .memory_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
                {
                    let ranges = [vk::MappedMemoryRange::default()
                        .memory(dst.allocation.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)];
                    dst.allocation
                        .inner
                        .device
                        .flush_mapped_memory_ranges(&ranges)
                        .map_err(|err| anyhow!("flushing shared host-copy memory: {err:?}"))?;
                }
                Ok::<(), anyhow::Error>(())
            };
            unsafe {
                self.allocation
                    .inner
                    .device
                    .unmap_memory(self.allocation.memory);
            }
            return copy_result;
        }

        let source_mapped = unsafe {
            self.allocation.inner.device.map_memory(
                self.allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|err| anyhow!("mapping Vulkan source memory for cross-device copy: {err:?}"))?;
        let destination_mapped = match unsafe {
            dst.allocation.inner.device.map_memory(
                dst.allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(mapped) => mapped,
            Err(err) => {
                unsafe {
                    self.allocation
                        .inner
                        .device
                        .unmap_memory(self.allocation.memory);
                }
                return Err(anyhow!(
                    "mapping Vulkan destination memory for cross-device copy: {err:?}"
                ));
            }
        };

        let copy_result = unsafe {
            if !self
                .allocation
                .memory_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                let ranges = [vk::MappedMemoryRange::default()
                    .memory(self.allocation.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                self.allocation
                    .inner
                    .device
                    .invalidate_mapped_memory_ranges(&ranges)
                    .map_err(|err| anyhow!("invalidating cross-device source memory: {err:?}"))?;
            }
            std::ptr::copy_nonoverlapping(
                source_mapped
                    .cast::<u8>()
                    .add(self.allocation.memory_offset)
                    .add(self.allocation.buffer_offset_bytes),
                destination_mapped
                    .cast::<u8>()
                    .add(dst.allocation.memory_offset)
                    .add(dst.allocation.buffer_offset_bytes),
                bytes,
            );
            if !dst
                .allocation
                .memory_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                let ranges = [vk::MappedMemoryRange::default()
                    .memory(dst.allocation.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                dst.allocation
                    .inner
                    .device
                    .flush_mapped_memory_ranges(&ranges)
                    .map_err(|err| anyhow!("flushing cross-device destination memory: {err:?}"))?;
            }
            Ok::<(), anyhow::Error>(())
        };
        unsafe {
            dst.allocation
                .inner
                .device
                .unmap_memory(dst.allocation.memory);
            self.allocation
                .inner
                .device
                .unmap_memory(self.allocation.memory);
        }
        copy_result
    }

    pub fn is_device_local(&self) -> bool {
        self.allocation
            .memory_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
    }
}

impl Drop for GpuBufferAllocation {
    fn drop(&mut self) {
        if let Some(lease) = self.scratch_lease {
            let waits = self.inner.take_scratch_lease_timeline_uses(lease.lease_id);
            self.inner
                .retire_scratch_lease_after_timelines(lease, waits);
            return;
        }
        if self.recycle_on_drop {
            let waits = self.inner.take_recyclable_buffer_timeline_uses(self.buffer);
            self.inner.retire_recyclable_buffer_after_timelines(
                DetachedGpuBufferAllocation {
                    buffer: self.buffer,
                    memory: self.memory,
                    memory_offset: self.memory_offset,
                    allocation_span_bytes: self.allocation_span_bytes,
                    size_bytes: self.size_bytes,
                    memory_flags: self.memory_flags,
                    dedicated_memory: self.dedicated_memory,
                },
                waits,
            );
            return;
        }
        unsafe {
            self.inner.device.destroy_buffer(self.buffer, None);
        }
        if self.dedicated_memory {
            unsafe { self.inner.device.free_memory(self.memory, None) };
            return;
        }
        if let Ok(mut allocator) = self.inner.memory_allocator.lock() {
            allocator.free(
                &self.inner.device,
                self.memory,
                self.memory_offset,
                self.allocation_span_bytes,
                self.size_bytes,
            );
        }
    }
}

fn find_memory_type_prefer(
    device: &VulkanDevice,
    type_bits: u32,
    preferred: vk::MemoryPropertyFlags,
    fallback: vk::MemoryPropertyFlags,
) -> Result<(u32, vk::MemoryPropertyFlags)> {
    let properties = unsafe {
        device
            .inner
            .instance
            .get_physical_device_memory_properties(device.physical_device)
    };
    let mut fallback_match = None;
    for index in 0..properties.memory_type_count {
        let supported = type_bits & (1 << index) != 0;
        let flags = properties.memory_types[index as usize].property_flags;
        if supported && flags.contains(preferred) {
            return Ok((index, flags));
        }
        if supported && flags.contains(fallback) && fallback_match.is_none() {
            fallback_match = Some((index, flags));
        }
    }
    fallback_match.with_context(|| {
        format!("no Vulkan memory type satisfies preferred {preferred:?} or fallback {fallback:?}")
    })
}

fn copy_buffer(
    device: &VulkanDevice,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    size_bytes: usize,
) -> Result<()> {
    let commands = ComputeBatch::new(device)?;
    let regions = [vk::BufferCopy::default()
        .src_offset(src.allocation.buffer_offset_bytes as u64)
        .dst_offset(dst.allocation.buffer_offset_bytes as u64)
        .size(size_bytes as u64)];
    commands.inner.with_command_pool_host_access(|| unsafe {
        commands.inner.device.cmd_copy_buffer(
            commands.command_buffer,
            src.allocation.buffer,
            dst.allocation.buffer,
            &regions,
        );
    })?;
    commands.submit()
}

pub(crate) struct ComputeBatch {
    inner: Arc<DeviceInner>,
    device: VulkanDevice,
    command_buffer: vk::CommandBuffer,
    descriptor_pools: Vec<DescriptorPoolArenaChunk>,
    descriptor_set_cache: HashMap<DescriptorSetCacheKey, vk::DescriptorSet>,
    descriptor_sets_allocated: usize,
    dispatch_count: usize,
    shader_barrier_count: usize,
    pipeline_bind_count: usize,
    descriptor_bind_count: usize,
    push_constant_write_count: usize,
    upload_count: usize,
    upload_bytes: usize,
    upload_arena: UploadArenaStorage,
    recyclable_buffer_keepalives: HashMap<u64, GpuBuffer>,
    scratch_lease_keepalives: HashMap<u64, GpuBuffer>,
    bound_pipeline: Option<vk::Pipeline>,
    bound_descriptor_pipeline_layout: Option<PipelineLayoutSignature>,
    bound_descriptor_layout: Option<DescriptorLayoutSignature>,
    bound_descriptor_set: Option<vk::DescriptorSet>,
    bound_descriptor_buffers: Vec<BufferRegionKey>,
    pushed_constant_layout: Option<PipelineLayoutSignature>,
    pushed_constants: Vec<u8>,
    pending_shader_buffers: HashSet<BufferRegionKey>,
    pending_shader_reads: HashSet<BufferRegionKey>,
    pending_shader_writes: HashSet<BufferRegionKey>,
    dispatch_dependency_trace: Option<DispatchDependencyTrace>,
    kernel_timestamp_profile: Option<KernelTimestampProfile>,
    finished: bool,
}

/// An in-flight Vulkan compute batch whose command-buffer resources remain
/// alive until its queue-completion primitive is observed. Timeline-capable
/// devices use a monotonically increasing per-queue semaphore and retain fences
/// only as a compatibility fallback. Cross-device transports can therefore
/// overlap independent queues while keeping command-buffer ownership explicit.
pub(crate) struct SubmittedComputeBatch {
    inner: Arc<DeviceInner>,
    batch: Option<ComputeBatch>,
    completion: SubmissionCompletion,
    completed: bool,
}

/// Lightweight queue-progress dependency detached from a submitted batch.
/// Timeline-capable callers can keep this token after dropping the submission
/// object; transient Vulkan resources remain owned by the device retirement
/// arena until the same timeline value is observed.
#[derive(Clone)]
pub(crate) struct SubmissionTimelineWait {
    inner: Arc<DeviceInner>,
    semaphore: vk::Semaphore,
    value: u64,
}

/// Shared ownership for one device-group semaphore. Timeline retirement waits
/// can outlive the transport slot that originally published them, so the raw
/// Vulkan semaphore must remain alive until the dependent optimizer submission
/// has consumed its wait value.
struct DeviceGroupSemaphoreShared {
    inner: Arc<DeviceInner>,
    semaphore: vk::Semaphore,
    timeline: bool,
}

/// Semaphore used to hand work between physical devices that belong to one
/// Vulkan logical device. Device groups prefer a timeline semaphore so each
/// reusable transport slot can advance monotonically without consuming a
/// one-shot binary payload. Older drivers retain the binary behavior.
#[derive(Clone)]
pub(crate) struct DeviceGroupSemaphore {
    shared: Arc<DeviceGroupSemaphoreShared>,
}

/// Cloneable GPU-side dependency published by a device-group transport after a
/// source read has been submitted. Consumers can attach this directly to a
/// later queue submission without waiting for a host fence or Condvar wakeup.
#[derive(Clone)]
pub(crate) struct DeviceGroupTimelineWait {
    semaphore: DeviceGroupSemaphore,
    value: u64,
}

impl DeviceGroupSemaphore {
    pub(crate) fn new(source: &VulkanDevice, destination: &VulkanDevice) -> Result<Self> {
        if !source.shares_logical_device_with(destination)
            || !source.is_multi_physical_device_logical_device()
            || source.device_mask == destination.device_mask
        {
            bail!(
                "device-group semaphore requires two distinct views of one multi-physical-device Vulkan logical device"
            );
        }
        let timeline = source.inner.device_group_timeline_semaphore_enabled;
        let semaphore = if timeline {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            unsafe { source.inner.device.create_semaphore(&create_info, None) }
        } else {
            unsafe {
                source
                    .inner
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
        }
        .map_err(|err| anyhow!("creating Vulkan device-group semaphore: {err:?}"))?;
        Ok(Self {
            shared: Arc::new(DeviceGroupSemaphoreShared {
                inner: Arc::clone(&source.inner),
                semaphore,
                timeline,
            }),
        })
    }

    pub(crate) fn is_timeline(&self) -> bool {
        self.shared.timeline
    }

    pub(crate) fn timeline_wait(&self, value: u64) -> Result<DeviceGroupTimelineWait> {
        if !self.is_timeline() {
            bail!("device-group GPU retirement wait requires a timeline semaphore");
        }
        if value == 0 {
            bail!("device-group timeline semaphore values must be positive");
        }
        Ok(DeviceGroupTimelineWait {
            semaphore: self.clone(),
            value,
        })
    }

    fn same_semaphore(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }
}

impl Drop for DeviceGroupSemaphoreShared {
    fn drop(&mut self) {
        unsafe { self.inner.device.destroy_semaphore(self.semaphore, None) };
    }
}

impl DeviceGroupTimelineWait {
    /// Derive a later value on the same monotonic device-group timeline.
    /// Replica-state retirement reserves one contiguous value span per model
    /// generation, so an optimizer run ending at range `n` only needs the
    /// range-`n` value: signaling it proves every earlier range on this
    /// semaphore has also retired.
    pub(crate) fn advanced_by(&self, delta: usize) -> Result<Self> {
        let delta =
            u64::try_from(delta).context("device-group timeline wait offset exceeds u64")?;
        let value = self
            .value
            .checked_add(delta)
            .context("device-group timeline wait value overflow")?;
        self.semaphore.timeline_wait(value)
    }

    fn coalesce(waits: &[Self]) -> Vec<Self> {
        let mut coalesced = Vec::<Self>::with_capacity(waits.len());
        for wait in waits {
            if let Some(existing) = coalesced
                .iter_mut()
                .find(|existing| existing.semaphore.same_semaphore(&wait.semaphore))
            {
                existing.value = existing.value.max(wait.value);
            } else {
                coalesced.push(wait.clone());
            }
        }
        coalesced
    }
}

/// Pair of binary semaphore objects whose payload is shared through an opaque
/// platform handle across two independent Vulkan logical devices.
pub(crate) struct OpaqueExternalSemaphorePair {
    source_inner: Arc<DeviceInner>,
    destination_inner: Arc<DeviceInner>,
    source: vk::Semaphore,
    destination: vk::Semaphore,
}

impl OpaqueExternalSemaphorePair {
    pub(crate) fn new(source: &VulkanDevice, destination: &VulkanDevice) -> Result<Self> {
        if source.shares_logical_device_with(destination) {
            bail!("opaque external semaphore probe requires independent logical devices");
        }
        if !source.inner.opaque_external_transport_enabled
            || !destination.inner.opaque_external_transport_enabled
        {
            bail!("opaque external semaphore probe requires enabled platform external-semaphore support");
        }

        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(platform_external_semaphore_handle_type());
        let source_create = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
        let source_semaphore =
            unsafe { source.inner.device.create_semaphore(&source_create, None) }
                .map_err(|err| anyhow!("creating exportable Vulkan semaphore: {err:?}"))?;

        let destination_semaphore = match unsafe {
            destination
                .inner
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        } {
            Ok(semaphore) => semaphore,
            Err(err) => {
                unsafe {
                    source
                        .inner
                        .device
                        .destroy_semaphore(source_semaphore, None)
                };
                return Err(anyhow!("creating imported Vulkan semaphore: {err:?}"));
            }
        };

        #[cfg(target_os = "windows")]
        let import_result: Result<()> = (|| {
            let source_external = ash::khr::external_semaphore_win32::Device::new(
                &source.inner.instance,
                &source.inner.device,
            );
            let get_info = vk::SemaphoreGetWin32HandleInfoKHR::default()
                .semaphore(source_semaphore)
                .handle_type(platform_external_semaphore_handle_type());
            let raw_handle = unsafe { source_external.get_semaphore_win32_handle(&get_info) }
                .map_err(|err| {
                    anyhow!("exporting Vulkan opaque Win32 semaphore handle: {err:?}")
                })?;
            let handle = OwnedWin32Handle::new(raw_handle)?;
            let destination_external = ash::khr::external_semaphore_win32::Device::new(
                &destination.inner.instance,
                &destination.inner.device,
            );
            let import_info = vk::ImportSemaphoreWin32HandleInfoKHR::default()
                .semaphore(destination_semaphore)
                .flags(vk::SemaphoreImportFlags::empty())
                .handle_type(platform_external_semaphore_handle_type())
                .handle(handle.raw());
            unsafe { destination_external.import_semaphore_win32_handle(&import_info) }.map_err(
                |err| anyhow!("importing Vulkan opaque Win32 semaphore handle: {err:?}"),
            )?;
            Ok(())
        })();

        #[cfg(unix)]
        let import_result: Result<()> = (|| {
            let source_external = ash::khr::external_semaphore_fd::Device::new(
                &source.inner.instance,
                &source.inner.device,
            );
            let get_info = vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(source_semaphore)
                .handle_type(platform_external_semaphore_handle_type());
            let fd = unsafe { source_external.get_semaphore_fd(&get_info) }
                .map_err(|err| anyhow!("exporting Vulkan opaque semaphore fd: {err:?}"))?;
            if fd < 0 {
                bail!("Vulkan returned an invalid opaque semaphore fd {fd}");
            }
            let handle = unsafe { OwnedFd::from_raw_fd(fd) };
            let raw_fd = handle.into_raw_fd();
            let destination_external = ash::khr::external_semaphore_fd::Device::new(
                &destination.inner.instance,
                &destination.inner.device,
            );
            let import_info = vk::ImportSemaphoreFdInfoKHR::default()
                .semaphore(destination_semaphore)
                .flags(vk::SemaphoreImportFlags::empty())
                .handle_type(platform_external_semaphore_handle_type())
                .fd(raw_fd);
            match unsafe { destination_external.import_semaphore_fd(&import_info) } {
                Ok(()) => Ok(()),
                Err(err) => {
                    drop(unsafe { OwnedFd::from_raw_fd(raw_fd) });
                    Err(anyhow!("importing Vulkan opaque semaphore fd: {err:?}"))
                }
            }
        })();

        if let Err(err) = import_result {
            unsafe {
                destination
                    .inner
                    .device
                    .destroy_semaphore(destination_semaphore, None);
                source
                    .inner
                    .device
                    .destroy_semaphore(source_semaphore, None);
            }
            return Err(err);
        }

        Ok(Self {
            source_inner: Arc::clone(&source.inner),
            destination_inner: Arc::clone(&destination.inner),
            source: source_semaphore,
            destination: destination_semaphore,
        })
    }

    pub(crate) fn source_semaphore(&self) -> vk::Semaphore {
        self.source
    }

    pub(crate) fn destination_semaphore(&self) -> vk::Semaphore {
        self.destination
    }
}

impl Drop for OpaqueExternalSemaphorePair {
    fn drop(&mut self) {
        unsafe {
            self.destination_inner
                .device
                .destroy_semaphore(self.destination, None);
            self.source_inner
                .device
                .destroy_semaphore(self.source, None);
        }
    }
}

const DESCRIPTOR_ARENA_SETS_PER_POOL: u32 = 128;
const DESCRIPTOR_ARENA_STORAGE_DESCRIPTORS_PER_POOL: u32 = 2048;
const UPLOAD_ARENA_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const UPLOAD_ARENA_ALIGNMENT: usize = 16;
const KERNEL_PROFILE_DISPATCHES_PER_POOL: usize = 4096;
const KERNEL_PROFILE_TOP_SHADERS: usize = 24;
const KERNEL_PROFILE_ENV: &str = "HIERARCHOS_VULKAN_PROFILE_KERNELS";

struct DescriptorPoolArenaChunk {
    pool: vk::DescriptorPool,
    remaining_sets: u32,
    remaining_storage_descriptors: u32,
    storage_descriptor_capacity: u32,
}

struct UploadArenaChunk {
    buffer: GpuBuffer,
    used_bytes: usize,
}

pub(crate) struct PersistentUploadArena {
    chunks: Mutex<Vec<UploadArenaChunk>>,
    in_use: AtomicBool,
}

impl PersistentUploadArena {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            chunks: Mutex::new(Vec::new()),
            in_use: AtomicBool::new(false),
        })
    }

    fn acquire(&self) -> Result<()> {
        self.in_use
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| anyhow!("persistent Vulkan upload arena already has an active batch"))
    }

    fn release(&self) {
        self.in_use.store(false, Ordering::Release);
    }

    pub(crate) fn buffer_count(&self) -> Result<usize> {
        self.chunks
            .lock()
            .map(|chunks| chunks.len())
            .map_err(|_| anyhow!("persistent Vulkan upload arena lock was poisoned"))
    }
}

enum UploadArenaStorage {
    Local(Vec<UploadArenaChunk>),
    Persistent(Arc<PersistentUploadArena>),
}

struct DispatchDependencyTrace {
    pending_shader_read_owners: HashMap<BufferRegionKey, u64>,
    pending_shader_write_owners: HashMap<BufferRegionKey, u64>,
    edges: HashMap<(u64, u64, DispatchHazard), usize>,
}

struct KernelTimestampProfile {
    timestamp_period_ns: f64,
    timestamp_valid_bits: u32,
    report_diagnostics: bool,
    chunks: Vec<KernelTimestampQueryChunk>,
}

struct KernelTimestampQueryChunk {
    pool: vk::QueryPool,
    shader_signatures: Vec<u64>,
}

#[derive(Default)]
struct KernelTimestampStats {
    dispatches: usize,
    ticks: u128,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
enum DispatchHazard {
    ReadAfterWrite,
    WriteAfterRead,
    WriteAfterWrite,
}

impl DispatchHazard {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReadAfterWrite => "raw",
            Self::WriteAfterRead => "war",
            Self::WriteAfterWrite => "waw",
        }
    }
}

/// Structural signature for the descriptor ABI used by a compute kernel.
///
/// Vulkan descriptor set layouts do not need to have the same raw handle to be
/// compatible; they need to be identically defined. Hierarchos currently uses
/// one contiguous set of storage-buffer bindings starting at binding zero, so
/// the binding count fully defines that set-layout ABI.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct DescriptorLayoutSignature {
    binding_count: u32,
}

/// Structural signature for the pipeline-layout ABI relevant to descriptor
/// retention and push-constant state.
///
/// Descriptor bindings remain compatible across pipeline transitions only when
/// the set layouts and push-constant ranges are identically defined. Keeping a
/// semantic signature here lets the device interner share Vulkan layout handles
/// and lets the command recorder retain state across compatible kernels.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct PipelineLayoutSignature {
    descriptor: DescriptorLayoutSignature,
    push_constant_size: u32,
}

#[derive(Default)]
struct LayoutInterner {
    descriptor_layouts: HashMap<DescriptorLayoutSignature, vk::DescriptorSetLayout>,
    pipeline_layouts: HashMap<PipelineLayoutSignature, vk::PipelineLayout>,
}

fn intern_kernel_layouts(
    device: &VulkanDevice,
    descriptor_signature: DescriptorLayoutSignature,
    pipeline_signature: PipelineLayoutSignature,
) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout)> {
    let mut interner = device
        .inner
        .layout_interner
        .lock()
        .map_err(|_| anyhow!("Vulkan layout interner lock was poisoned"))?;

    let descriptor_set_layout = if let Some(layout) = interner
        .descriptor_layouts
        .get(&descriptor_signature)
        .copied()
    {
        layout
    } else {
        let bindings: Vec<_> = (0..descriptor_signature.binding_count)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe {
            device
                .inner
                .device
                .create_descriptor_set_layout(&layout_info, None)
        }
        .map_err(|err| anyhow!("creating interned Vulkan descriptor set layout: {err:?}"))?;
        interner
            .descriptor_layouts
            .insert(descriptor_signature, layout);
        layout
    };

    let pipeline_layout =
        if let Some(layout) = interner.pipeline_layouts.get(&pipeline_signature).copied() {
            layout
        } else {
            let set_layouts = [descriptor_set_layout];
            let push_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pipeline_signature.push_constant_size)];
            let mut pipeline_layout_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if pipeline_signature.push_constant_size != 0 {
                pipeline_layout_info = pipeline_layout_info.push_constant_ranges(&push_ranges);
            }
            let layout = unsafe {
                device
                    .inner
                    .device
                    .create_pipeline_layout(&pipeline_layout_info, None)
            }
            .map_err(|err| anyhow!("creating interned Vulkan pipeline layout: {err:?}"))?;
            interner.pipeline_layouts.insert(pipeline_signature, layout);
            layout
        };

    Ok((descriptor_set_layout, pipeline_layout))
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct BufferRegionKey {
    buffer: u64,
    offset: u64,
    size: u64,
}

#[derive(Hash, Eq, PartialEq)]
struct DescriptorSetCacheKey {
    layout: DescriptorLayoutSignature,
    buffers: Vec<BufferRegionKey>,
}

impl ComputeBatch {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Self::new_with_upload_storage(device, UploadArenaStorage::Local(Vec::new()))
    }

    pub(crate) fn new_with_persistent_upload_arena(
        device: &VulkanDevice,
        upload_arena: Arc<PersistentUploadArena>,
    ) -> Result<Self> {
        upload_arena.acquire()?;
        {
            let mut chunks = match upload_arena.chunks.lock() {
                Ok(chunks) => chunks,
                Err(_) => {
                    upload_arena.release();
                    bail!("persistent Vulkan upload arena lock was poisoned");
                }
            };
            for chunk in chunks.iter_mut() {
                chunk.used_bytes = 0;
            }
        }
        let batch = Self::new_with_upload_storage(
            device,
            UploadArenaStorage::Persistent(Arc::clone(&upload_arena)),
        );
        if batch.is_err() {
            upload_arena.release();
        }
        batch
    }

    fn new_with_upload_storage(
        device: &VulkanDevice,
        upload_arena: UploadArenaStorage,
    ) -> Result<Self> {
        let command_buffer = device.inner.acquire_compute_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(err) = device.inner.with_command_pool_host_access(|| unsafe {
            device
                .inner
                .device
                .begin_command_buffer(command_buffer, &begin)
        })? {
            device.inner.recycle_compute_command_buffer(command_buffer);
            return Err(anyhow!("beginning Vulkan compute batch: {err:?}"));
        }
        if device.is_multi_physical_device_logical_device() {
            device.inner.with_command_pool_host_access(|| unsafe {
                device
                    .inner
                    .device
                    .cmd_set_device_mask(command_buffer, device.device_mask);
            })?;
        }
        let explicit_kernel_profile = std::env::var_os(KERNEL_PROFILE_ENV).is_some();
        let scheduler_kernel_profile = device
            .inner
            .scheduler_kernel_timestamp_collection_enabled
            .load(Ordering::Acquire);
        let kernel_timestamp_profile = if explicit_kernel_profile || scheduler_kernel_profile {
            let queue_families = unsafe {
                device
                    .inner
                    .instance
                    .get_physical_device_queue_family_properties(device.physical_device)
            };
            let queue_family = queue_families
                .get(device.inner.queue_family_index as usize)
                .with_context(|| {
                    format!(
                        "selected Vulkan queue family {} disappeared while enabling kernel profiling",
                        device.inner.queue_family_index
                    )
                })?;
            if queue_family.timestamp_valid_bits == 0 {
                if explicit_kernel_profile {
                    device.inner.recycle_compute_command_buffer(command_buffer);
                    bail!(
                        "{KERNEL_PROFILE_ENV}=1 requires compute timestamp queries, but Vulkan device {:?} reports timestampValidBits=0 for queue family {}",
                        device.name(),
                        device.inner.queue_family_index
                    );
                }
                None
            } else {
                let properties = unsafe {
                    device
                        .inner
                        .instance
                        .get_physical_device_properties(device.physical_device)
                };
                Some(KernelTimestampProfile {
                    timestamp_period_ns: f64::from(properties.limits.timestamp_period),
                    timestamp_valid_bits: queue_family.timestamp_valid_bits,
                    report_diagnostics: explicit_kernel_profile,
                    chunks: Vec::new(),
                })
            }
        } else {
            None
        };
        Ok(Self {
            inner: Arc::clone(&device.inner),
            device: device.clone(),
            command_buffer,
            descriptor_pools: Vec::new(),
            descriptor_set_cache: HashMap::new(),
            descriptor_sets_allocated: 0,
            dispatch_count: 0,
            shader_barrier_count: 0,
            pipeline_bind_count: 0,
            descriptor_bind_count: 0,
            push_constant_write_count: 0,
            upload_count: 0,
            upload_bytes: 0,
            upload_arena,
            recyclable_buffer_keepalives: HashMap::new(),
            scratch_lease_keepalives: HashMap::new(),
            bound_pipeline: None,
            bound_descriptor_pipeline_layout: None,
            bound_descriptor_layout: None,
            bound_descriptor_set: None,
            bound_descriptor_buffers: Vec::new(),
            pushed_constant_layout: None,
            pushed_constants: Vec::new(),
            pending_shader_buffers: HashSet::new(),
            pending_shader_reads: HashSet::new(),
            pending_shader_writes: HashSet::new(),
            dispatch_dependency_trace: std::env::var_os("HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS")
                .map(|_| DispatchDependencyTrace {
                    pending_shader_read_owners: HashMap::new(),
                    pending_shader_write_owners: HashMap::new(),
                    edges: HashMap::new(),
                }),
            kernel_timestamp_profile,
            finished: false,
        })
    }

    pub(crate) fn upload_f32(&mut self, dst: &GpuBuffer, values: &[f32]) -> Result<()> {
        if values.iter().any(|value| !value.is_finite()) {
            bail!("refusing to upload non-finite FP32 values");
        }
        self.upload_bytes(dst, bytemuck::cast_slice(values))
    }

    pub(crate) fn upload_u32(&mut self, dst: &GpuBuffer, values: &[u32]) -> Result<()> {
        self.upload_bytes(dst, bytemuck::cast_slice(values))
    }

    fn upload_bytes(&mut self, dst: &GpuBuffer, bytes: &[u8]) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &dst.allocation.inner) {
            bail!("upload destination and compute batch belong to different Vulkan devices");
        }
        if bytes.is_empty() || bytes.len() > dst.allocation.size_bytes {
            bail!(
                "batch upload size {} is outside destination capacity 1..={}",
                bytes.len(),
                dst.allocation.size_bytes
            );
        }
        self.retain_recyclable_buffers(&[dst]);
        // Always stage batch uploads, even when the destination allocation is
        // host-visible. A command batch can record several writes to the same
        // scratch buffer before submission; directly mapping the destination
        // would let a later host write overwrite data that an earlier recorded
        // dispatch has not consumed yet. Staging turns those writes into ordered
        // Vulkan copies and makes single-submit replay correct on integrated
        // GPUs where DEVICE_LOCAL memory is commonly HOST_VISIBLE as well.
        let (staging_buffer, staging_offset) = self.stage_upload(bytes)?;
        let regions = [vk::BufferCopy::default()
            .src_offset(staging_offset as u64)
            .dst_offset(dst.allocation.buffer_offset_bytes as u64)
            .size(bytes.len() as u64)];
        self.inner.with_command_pool_host_access(|| unsafe {
            self.inner.device.cmd_copy_buffer(
                self.command_buffer,
                staging_buffer,
                dst.allocation.buffer,
                &regions,
            );
            let barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(dst.allocation.buffer)
                .offset(dst.allocation.buffer_offset_bytes as u64)
                .size(bytes.len() as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        })?;
        self.upload_count = self
            .upload_count
            .checked_add(1)
            .context("Vulkan compute-batch upload count overflow")?;
        self.upload_bytes = self
            .upload_bytes
            .checked_add(bytes.len())
            .context("Vulkan compute-batch upload byte count overflow")?;
        Ok(())
    }

    fn stage_upload(&mut self, bytes: &[u8]) -> Result<(vk::Buffer, usize)> {
        match &mut self.upload_arena {
            UploadArenaStorage::Local(chunks) => {
                stage_upload_in_chunks(&self.device, chunks, bytes)
            }
            UploadArenaStorage::Persistent(arena) => {
                let mut chunks = arena
                    .chunks
                    .lock()
                    .map_err(|_| anyhow!("persistent Vulkan upload arena lock was poisoned"))?;
                stage_upload_in_chunks(&self.device, &mut chunks, bytes)
            }
        }
    }

    pub(crate) fn descriptor_pool_count(&self) -> usize {
        self.descriptor_pools.len()
    }

    fn uses_persistent_upload_arena(&self) -> bool {
        matches!(self.upload_arena, UploadArenaStorage::Persistent(_))
    }

    fn retain_recyclable_buffers(&mut self, buffers: &[&GpuBuffer]) {
        for buffer in buffers {
            if let Some(lease) = buffer.allocation.scratch_lease {
                self.scratch_lease_keepalives
                    .entry(lease.lease_id)
                    .or_insert_with(|| (*buffer).clone());
                continue;
            }
            if !buffer.allocation.recycle_on_drop {
                continue;
            }
            self.recyclable_buffer_keepalives
                .entry(buffer.allocation.buffer.as_raw())
                .or_insert_with(|| (*buffer).clone());
        }
    }

    fn retire_recyclable_buffers_on_timeline(&mut self, semaphore: vk::Semaphore, value: u64) {
        for buffer in self.recyclable_buffer_keepalives.values() {
            self.inner.register_recyclable_buffer_timeline_use(
                buffer.allocation.buffer,
                semaphore,
                value,
            );
        }
        self.recyclable_buffer_keepalives.clear();
        for buffer in self.scratch_lease_keepalives.values() {
            if let Some(lease) = buffer.allocation.scratch_lease {
                self.inner
                    .register_scratch_lease_timeline_use(lease.lease_id, semaphore, value);
            }
        }
        self.scratch_lease_keepalives.clear();
    }

    fn detach_timeline_submission_resources(
        &mut self,
        semaphore: vk::Semaphore,
        value: u64,
    ) -> Result<()> {
        let local_upload_allocations = match &self.upload_arena {
            UploadArenaStorage::Local(chunks) => {
                if chunks
                    .iter()
                    .any(|chunk| Arc::strong_count(&chunk.buffer.allocation) != 1)
                {
                    bail!(
                        "local Vulkan upload arena unexpectedly contains a shared staging buffer"
                    );
                }
                let chunks = match std::mem::replace(
                    &mut self.upload_arena,
                    UploadArenaStorage::Local(Vec::new()),
                ) {
                    UploadArenaStorage::Local(chunks) => chunks,
                    UploadArenaStorage::Persistent(_) => unreachable!(),
                };
                chunks
                    .into_iter()
                    .map(|chunk| chunk.buffer.into_detached_allocation())
                    .collect()
            }
            UploadArenaStorage::Persistent(_) => Vec::new(),
        };
        let command_buffer = std::mem::replace(&mut self.command_buffer, vk::CommandBuffer::null());
        let descriptor_pools = std::mem::take(&mut self.descriptor_pools);
        let kernel_timestamp_profile = self.kernel_timestamp_profile.take();
        self.inner.retire_submission_resources_on_timeline(
            command_buffer,
            descriptor_pools,
            local_upload_allocations,
            kernel_timestamp_profile,
            semaphore,
            value,
        );
        Ok(())
    }
}

fn stage_upload_in_chunks(
    device: &VulkanDevice,
    chunks: &mut Vec<UploadArenaChunk>,
    bytes: &[u8],
) -> Result<(vk::Buffer, usize)> {
    let aligned_len = align_up(bytes.len(), UPLOAD_ARENA_ALIGNMENT)
        .context("Vulkan upload arena size overflow")?;
    let can_reuse = chunks.last().is_some_and(|chunk| {
        align_up(chunk.used_bytes, UPLOAD_ARENA_ALIGNMENT)
            .and_then(|offset| offset.checked_add(bytes.len()))
            .is_some_and(|end| end <= chunk.buffer.allocation.size_bytes)
    });
    if !can_reuse {
        let capacity = UPLOAD_ARENA_CHUNK_BYTES.max(aligned_len);
        chunks.push(UploadArenaChunk {
            buffer: GpuBuffer::new_recyclable_host_visible(device, capacity)?,
            used_bytes: 0,
        });
    }
    let chunk = chunks
        .last_mut()
        .context("Vulkan upload arena failed to retain its staging chunk")?;
    let offset = align_up(chunk.used_bytes, UPLOAD_ARENA_ALIGNMENT)
        .context("Vulkan upload arena offset overflow")?;
    chunk.buffer.write_mapped_range(offset, bytes)?;
    chunk.used_bytes = offset
        .checked_add(bytes.len())
        .context("Vulkan upload arena offset overflow")?;
    Ok((chunk.buffer.allocation.buffer, offset))
}

impl ComputeBatch {
    pub(crate) fn descriptor_set_count(&self) -> usize {
        self.descriptor_sets_allocated
    }

    pub(crate) fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    pub(crate) fn shader_barrier_count(&self) -> usize {
        self.shader_barrier_count
    }

    pub(crate) fn pipeline_bind_count(&self) -> usize {
        self.pipeline_bind_count
    }

    pub(crate) fn descriptor_bind_count(&self) -> usize {
        self.descriptor_bind_count
    }

    pub(crate) fn push_constant_write_count(&self) -> usize {
        self.push_constant_write_count
    }

    pub(crate) fn upload_count(&self) -> usize {
        self.upload_count
    }

    pub(crate) fn uploaded_bytes(&self) -> usize {
        self.upload_bytes
    }

    pub(crate) fn upload_arena_buffer_count(&self) -> usize {
        match &self.upload_arena {
            UploadArenaStorage::Local(chunks) => chunks.len(),
            UploadArenaStorage::Persistent(arena) => arena.buffer_count().unwrap_or_default(),
        }
    }

    pub(crate) fn readback_f32(
        &mut self,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        self.readback_f32_range(src, 0, dst, 0, len)
    }

    pub(crate) fn readback_f32_range(
        &mut self,
        src: &GpuBuffer,
        src_offset: usize,
        dst: &GpuBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &src.allocation.inner)
            || !Arc::ptr_eq(&self.inner, &dst.allocation.inner)
        {
            bail!("readback buffers and compute batch belong to different Vulkan devices");
        }
        if !dst
            .allocation
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            bail!("readback destination must be host-visible");
        }
        let bytes = len.checked_mul(4).context("readback size overflow")?;
        let src_offset_bytes = src_offset
            .checked_mul(4)
            .context("readback source offset overflow")?;
        let dst_offset_bytes = dst_offset
            .checked_mul(4)
            .context("readback destination offset overflow")?;
        let src_end = src_offset_bytes
            .checked_add(bytes)
            .context("readback source range overflow")?;
        let dst_end = dst_offset_bytes
            .checked_add(bytes)
            .context("readback destination range overflow")?;
        if bytes == 0 || src_end > src.allocation.size_bytes || dst_end > dst.allocation.size_bytes
        {
            bail!(
                "readback ranges src={src_offset_bytes}..{src_end} dst={dst_offset_bytes}..{dst_end} exceed source/destination capacities {}/{}",
                src.allocation.size_bytes,
                dst.allocation.size_bytes
            );
        }
        let src_buffer_offset_bytes = src.absolute_buffer_offset(src_offset_bytes)?;
        let dst_buffer_offset_bytes = dst.absolute_buffer_offset(dst_offset_bytes)?;
        self.retain_recyclable_buffers(&[src, dst]);
        self.inner.with_command_pool_host_access(|| unsafe {
            let src_barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(src.allocation.buffer)
                .offset(src_buffer_offset_bytes as u64)
                .size(bytes as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &src_barriers,
                &[],
            );
            let regions = [vk::BufferCopy::default()
                .src_offset(src_buffer_offset_bytes as u64)
                .dst_offset(dst_buffer_offset_bytes as u64)
                .size(bytes as u64)];
            self.inner.device.cmd_copy_buffer(
                self.command_buffer,
                src.allocation.buffer,
                dst.allocation.buffer,
                &regions,
            );
            let dst_barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(dst.allocation.buffer)
                .offset(dst_buffer_offset_bytes as u64)
                .size(bytes as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &dst_barriers,
                &[],
            );
        })?;
        Ok(())
    }

    pub(crate) fn copy_f32(&mut self, src: &GpuBuffer, dst: &GpuBuffer, len: usize) -> Result<()> {
        self.copy_f32_range(src, 0, dst, 0, len)
    }

    pub(crate) fn copy_f32_range(
        &mut self,
        src: &GpuBuffer,
        src_offset: usize,
        dst: &GpuBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &src.allocation.inner)
            || !Arc::ptr_eq(&self.inner, &dst.allocation.inner)
        {
            bail!("copy buffers and compute batch belong to different Vulkan devices");
        }
        let bytes = len.checked_mul(4).context("device copy size overflow")?;
        let src_offset_bytes = src_offset
            .checked_mul(4)
            .context("device copy source offset overflow")?;
        let dst_offset_bytes = dst_offset
            .checked_mul(4)
            .context("device copy destination offset overflow")?;
        let src_end = src_offset_bytes
            .checked_add(bytes)
            .context("device copy source range overflow")?;
        let dst_end = dst_offset_bytes
            .checked_add(bytes)
            .context("device copy destination range overflow")?;
        if bytes == 0 || src_end > src.allocation.size_bytes || dst_end > dst.allocation.size_bytes
        {
            bail!(
                "device copy ranges src={src_offset_bytes}..{src_end} dst={dst_offset_bytes}..{dst_end} exceed source/destination capacities {}/{}",
                src.allocation.size_bytes,
                dst.allocation.size_bytes
            );
        }
        let src_buffer_offset_bytes = src.absolute_buffer_offset(src_offset_bytes)?;
        let dst_buffer_offset_bytes = dst.absolute_buffer_offset(dst_offset_bytes)?;
        if src.allocation.buffer == dst.allocation.buffer
            && src_buffer_offset_bytes == dst_buffer_offset_bytes
        {
            return Ok(());
        }
        self.retain_recyclable_buffers(&[src, dst]);
        self.inner.with_command_pool_host_access(|| unsafe {
            let pre_barriers = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::HOST_WRITE
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(src.allocation.buffer)
                    .offset(src_buffer_offset_bytes as u64)
                    .size(bytes as u64),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(dst.allocation.buffer)
                    .offset(dst_buffer_offset_bytes as u64)
                    .size(bytes as u64),
            ];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &pre_barriers,
                &[],
            );
            let regions = [vk::BufferCopy::default()
                .src_offset(src_buffer_offset_bytes as u64)
                .dst_offset(dst_buffer_offset_bytes as u64)
                .size(bytes as u64)];
            self.inner.device.cmd_copy_buffer(
                self.command_buffer,
                src.allocation.buffer,
                dst.allocation.buffer,
                &regions,
            );
            let post_barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ,
                )
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(dst.allocation.buffer)
                .offset(dst_buffer_offset_bytes as u64)
                .size(bytes as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &post_barriers,
                &[],
            );
        })?;
        Ok(())
    }

    pub(crate) fn release_buffer_to_external(
        &mut self,
        buffer: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &buffer.allocation.inner) {
            bail!("external-memory release buffer and compute batch belong to different Vulkan devices");
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .context("external-memory release size overflow")?;
        if bytes == 0 || bytes > buffer.allocation.size_bytes {
            bail!(
                "external-memory release size {bytes} is outside buffer capacity 1..={}",
                buffer.allocation.size_bytes
            );
        }
        let barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::HOST_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::empty())
            .src_queue_family_index(self.inner.queue_family_index)
            .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .buffer(buffer.allocation.buffer)
            .offset(buffer.allocation.buffer_offset_bytes as u64)
            .size(bytes as u64)];
        self.inner.with_command_pool_host_access(|| unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &barrier,
                &[],
            );
        })?;
        Ok(())
    }

    pub(crate) fn acquire_buffer_from_external(
        &mut self,
        buffer: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &buffer.allocation.inner) {
            bail!("external-memory acquire buffer and compute batch belong to different Vulkan devices");
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .context("external-memory acquire size overflow")?;
        if bytes == 0 || bytes > buffer.allocation.size_bytes {
            bail!(
                "external-memory acquire size {bytes} is outside buffer capacity 1..={}",
                buffer.allocation.size_bytes
            );
        }
        let barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(self.inner.queue_family_index)
            .buffer(buffer.allocation.buffer)
            .offset(buffer.allocation.buffer_offset_bytes as u64)
            .size(bytes as u64)];
        self.inner.with_command_pool_host_access(|| unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &barrier,
                &[],
            );
        })?;
        Ok(())
    }

    pub(crate) fn fill_f32(&mut self, dst: &GpuBuffer, len: usize, value: f32) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &dst.allocation.inner) {
            bail!("fill destination and compute batch belong to different Vulkan devices");
        }
        let bytes = len.checked_mul(4).context("device fill size overflow")?;
        if bytes == 0 || bytes > dst.allocation.size_bytes {
            bail!(
                "device fill of {bytes} bytes exceeds destination capacity {}",
                dst.allocation.size_bytes
            );
        }
        self.retain_recyclable_buffers(&[dst]);
        self.inner.with_command_pool_host_access(|| unsafe {
            let pre_barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(dst.allocation.buffer)
                .offset(dst.allocation.buffer_offset_bytes as u64)
                .size(bytes as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &pre_barriers,
                &[],
            );
            self.inner.device.cmd_fill_buffer(
                self.command_buffer,
                dst.allocation.buffer,
                dst.allocation.buffer_offset_bytes as u64,
                bytes as u64,
                value.to_bits(),
            );
            let post_barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(dst.allocation.buffer)
                .offset(dst.allocation.buffer_offset_bytes as u64)
                .size(bytes as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &post_barriers,
                &[],
            );
        })?;
        Ok(())
    }

    pub(crate) fn fill_zero_f32(&mut self, dst: &GpuBuffer, len: usize) -> Result<()> {
        self.fill_f32(dst, len, 0.0)
    }

    pub(crate) fn update_f32_at(
        &mut self,
        dst: &GpuBuffer,
        index: usize,
        value: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &dst.allocation.inner) {
            bail!("update destination and compute batch belong to different Vulkan devices");
        }
        if !value.is_finite() {
            bail!("refusing to update a non-finite FP32 value");
        }
        let relative_offset = index
            .checked_mul(std::mem::size_of::<f32>())
            .context("device update offset overflow")?;
        let end = relative_offset
            .checked_add(std::mem::size_of::<f32>())
            .context("device update range overflow")?;
        if end > dst.allocation.size_bytes {
            bail!(
                "device FP32 update at index {index} exceeds destination capacity {} bytes",
                dst.allocation.size_bytes
            );
        }
        let absolute_offset = dst
            .allocation
            .buffer_offset_bytes
            .checked_add(relative_offset)
            .context("device update absolute offset overflow")?;
        self.retain_recyclable_buffers(&[dst]);
        let bytes = value.to_ne_bytes();
        self.inner.with_command_pool_host_access(|| unsafe {
            let pre_barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(dst.allocation.buffer)
                .offset(absolute_offset as u64)
                .size(bytes.len() as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &pre_barriers,
                &[],
            );
            self.inner.device.cmd_update_buffer(
                self.command_buffer,
                dst.allocation.buffer,
                absolute_offset as u64,
                &bytes,
            );
            let post_barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(dst.allocation.buffer)
                .offset(absolute_offset as u64)
                .size(bytes.len() as u64)];
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &post_barriers,
                &[],
            );
        })?;
        Ok(())
    }

    fn allocate_descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
        binding_count: usize,
    ) -> Result<vk::DescriptorSet> {
        let required_descriptors = u32::try_from(binding_count)
            .context("descriptor binding count exceeds Vulkan u32 range")?;
        let needs_pool = self.descriptor_pools.last().is_none_or(|chunk| {
            chunk.remaining_sets == 0 || chunk.remaining_storage_descriptors < required_descriptors
        });
        if needs_pool {
            self.descriptor_pools.push(
                self.inner
                    .acquire_descriptor_pool_chunk(required_descriptors)?,
            );
        }

        let chunk = self
            .descriptor_pools
            .last_mut()
            .context("descriptor arena failed to retain its allocation pool")?;
        let set_layouts = [layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(chunk.pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { self.inner.device.allocate_descriptor_sets(&allocate_info) }
            .map_err(|err| anyhow!("allocating Vulkan descriptor arena set: {err:?}"))?[0];
        chunk.remaining_sets -= 1;
        chunk.remaining_storage_descriptors -= required_descriptors;
        self.descriptor_sets_allocated = self
            .descriptor_sets_allocated
            .checked_add(1)
            .context("descriptor arena set-count overflow")?;
        Ok(descriptor_set)
    }

    fn descriptor_set_for(
        &mut self,
        layout: vk::DescriptorSetLayout,
        layout_signature: DescriptorLayoutSignature,
        binding_count: usize,
        buffers: &[&GpuBuffer],
    ) -> Result<vk::DescriptorSet> {
        let key = DescriptorSetCacheKey {
            layout: layout_signature,
            buffers: buffers
                .iter()
                .map(|buffer| buffer.buffer_region_key())
                .collect(),
        };
        if let Some(descriptor_set) = self.descriptor_set_cache.get(&key) {
            return Ok(*descriptor_set);
        }

        let descriptor_set = self.allocate_descriptor_set(layout, binding_count)?;
        let infos: Vec<_> = buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.allocation.buffer)
                    .offset(buffer.allocation.buffer_offset_bytes as u64)
                    .range(buffer.allocation.size_bytes as u64)
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.inner.device.update_descriptor_sets(&writes, &[]) };
        self.descriptor_set_cache.insert(key, descriptor_set);
        Ok(descriptor_set)
    }

    pub(crate) fn submit(self) -> Result<()> {
        self.submit_async()?.wait()
    }

    /// Submit the current command stream synchronously and immediately replace
    /// it with a fresh command buffer that keeps the same upload-arena policy.
    /// This is intended for rare graph phase boundaries where a small host
    /// scalar decision depends on Vulkan forward readback before reverse-mode
    /// commands can be recorded (for example sequence-scalar backward caps).
    pub(crate) fn submit_and_restart(&mut self) -> Result<()> {
        let device = self.device.clone();
        let persistent_arena = match &self.upload_arena {
            UploadArenaStorage::Persistent(arena) => Some(Arc::clone(arena)),
            UploadArenaStorage::Local(_) => None,
        };

        // A persistent upload arena is exclusively acquired by a live batch,
        // so install a short-lived local placeholder before submitting `current`.
        // Once submission/wait releases the arena, the replacement can acquire
        // that same arena for the reverse phase.
        let placeholder = Self::new(&device)?;
        let current = std::mem::replace(self, placeholder);
        current.submit()?;
        *self = match persistent_arena {
            Some(arena) => Self::new_with_persistent_upload_arena(&device, arena)?,
            None => Self::new(&device)?,
        };
        Ok(())
    }

    pub(crate) fn submit_async(self) -> Result<SubmittedComputeBatch> {
        self.submit_async_with_device_group_semaphores_and_submission_waits(&[], &[], &[])
    }

    pub(crate) fn submit_async_signal_raw_binary_semaphore(
        self,
        semaphore: vk::Semaphore,
    ) -> Result<SubmittedComputeBatch> {
        self.submit_async_with_raw_binary_semaphores(None, Some(semaphore))
    }

    pub(crate) fn submit_async_wait_raw_binary_semaphore(
        self,
        semaphore: vk::Semaphore,
    ) -> Result<SubmittedComputeBatch> {
        self.submit_async_with_raw_binary_semaphores(Some(semaphore), None)
    }

    pub(crate) fn submit_async_wait_signal_raw_binary_semaphores(
        self,
        wait: vk::Semaphore,
        signal: vk::Semaphore,
    ) -> Result<SubmittedComputeBatch> {
        self.submit_async_with_raw_binary_semaphores(Some(wait), Some(signal))
    }

    fn submit_async_with_raw_binary_semaphores(
        self,
        wait: Option<vk::Semaphore>,
        signal: Option<vk::Semaphore>,
    ) -> Result<SubmittedComputeBatch> {
        if self.device.is_multi_physical_device_logical_device() {
            bail!("raw external semaphore submission is only valid for an independent single-physical-device logical device");
        }
        if wait == Some(vk::Semaphore::null()) || signal == Some(vk::Semaphore::null()) {
            bail!("raw external semaphore submission received a null semaphore");
        }

        self.inner
            .with_command_pool_host_access(|| unsafe {
                self.inner.device.end_command_buffer(self.command_buffer)
            })?
            .map_err(|err| anyhow!("ending Vulkan compute batch: {err:?}"))?;
        let completion = {
            let queue_index = self.device.queue_index;
            let queue = self.inner.queues[queue_index];
            let _queue_guard = self.inner.queue_locks[queue_index]
                .lock()
                .map_err(|_| anyhow!("Vulkan queue lock was poisoned"))?;
            let timeline = self.inner.reserve_submission_timeline(queue_index)?;
            let fence = if timeline.is_some() {
                vk::Fence::null()
            } else {
                unsafe {
                    self.inner
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                }
                .map_err(|err| anyhow!("creating Vulkan compute-batch fence: {err:?}"))?
            };
            let command_buffers = [self.command_buffer];
            let wait_semaphores = wait.into_iter().collect::<Vec<_>>();
            let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_semaphores.len()];
            let mut signal_semaphores = signal.into_iter().collect::<Vec<_>>();
            let wait_values = vec![0u64; wait_semaphores.len()];
            let mut signal_values = vec![0u64; signal_semaphores.len()];
            if let Some((semaphore, value)) = timeline {
                signal_semaphores.push(semaphore);
                signal_values.push(value);
            }
            let mut timeline_submit = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let mut submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            if timeline.is_some() {
                submit = submit.push_next(&mut timeline_submit);
            }
            let submits = [submit];
            if let Err(err) = unsafe { self.inner.device.queue_submit(queue, &submits, fence) } {
                if fence != vk::Fence::null() {
                    unsafe { self.inner.device.destroy_fence(fence, None) };
                }
                return Err(anyhow!(
                    "submitting Vulkan compute batch with external semaphore: {err:?}"
                ));
            }
            match timeline {
                Some((semaphore, value)) => SubmissionCompletion::Timeline { semaphore, value },
                None => SubmissionCompletion::Fence(fence),
            }
        };
        self.into_submitted(completion)
    }

    pub(crate) fn submit_async_signal_device_group(
        self,
        semaphore: &DeviceGroupSemaphore,
        value: u64,
    ) -> Result<SubmittedComputeBatch> {
        let signals = [(semaphore, value)];
        self.submit_async_with_device_group_semaphores_and_submission_waits(&[], &[], &signals)
    }

    pub(crate) fn submit_async_wait_signal_device_group(
        self,
        wait_semaphore: &DeviceGroupSemaphore,
        wait_value: u64,
        signal_semaphore: &DeviceGroupSemaphore,
        signal_value: u64,
    ) -> Result<SubmittedComputeBatch> {
        let waits = [(wait_semaphore, wait_value)];
        let signals = [(signal_semaphore, signal_value)];
        self.submit_async_with_device_group_semaphores_and_submission_waits(&[], &waits, &signals)
    }

    /// Submit one device-group batch with arbitrary semaphore waits/signals.
    /// Replica-state broadcast uses this to signal both the reusable transport
    /// handoff and a separately persistent AdamW-retirement timeline value from
    /// the same source-copy submission.
    pub(crate) fn submit_async_device_group(
        self,
        waits: &[(&DeviceGroupSemaphore, u64)],
        signals: &[(&DeviceGroupSemaphore, u64)],
    ) -> Result<SubmittedComputeBatch> {
        self.submit_async_with_device_group_semaphores_and_submission_waits(&[], waits, signals)
    }

    /// Submit a device-group batch behind a previously submitted queue timeline
    /// value as well as the ordinary device-group handoff dependencies. This is
    /// the bridge that lets a new optimizer generation become CPU-visible as soon
    /// as its mutation tail is queued while independent replica source queues
    /// still wait for the exact GPU completion point before reading that state.
    pub(crate) fn submit_async_device_group_after_submission_timeline(
        self,
        submission_waits: &[SubmissionTimelineWait],
        waits: &[(&DeviceGroupSemaphore, u64)],
        signals: &[(&DeviceGroupSemaphore, u64)],
    ) -> Result<SubmittedComputeBatch> {
        self.submit_async_with_device_group_semaphores_and_submission_waits(
            submission_waits,
            waits,
            signals,
        )
    }

    /// Submit this batch behind one or more timeline values published by
    /// device-group transport. Duplicate semaphore waits are collapsed to the
    /// largest value so a coalesced AdamW range run emits the minimal Vulkan
    /// dependency set.
    pub(crate) fn submit_async_wait_device_group_timeline(
        self,
        waits: &[DeviceGroupTimelineWait],
    ) -> Result<SubmittedComputeBatch> {
        if waits.is_empty() {
            return self.submit_async();
        }
        let waits = DeviceGroupTimelineWait::coalesce(waits);
        let wait_refs = waits
            .iter()
            .map(|wait| (&wait.semaphore, wait.value))
            .collect::<Vec<_>>();
        self.submit_async_with_device_group_semaphores_and_submission_waits(&[], &wait_refs, &[])
    }

    fn submit_async_with_device_group_semaphores_and_submission_waits(
        self,
        submission_waits: &[SubmissionTimelineWait],
        waits: &[(&DeviceGroupSemaphore, u64)],
        signals: &[(&DeviceGroupSemaphore, u64)],
    ) -> Result<SubmittedComputeBatch> {
        if !waits.is_empty() || !signals.is_empty() {
            if !self.device.is_multi_physical_device_logical_device() {
                bail!("device-group semaphore submission requires a multi-physical-device logical device");
            }
            for (semaphore, value) in waits.iter().copied().chain(signals.iter().copied()) {
                if !Arc::ptr_eq(&self.inner, &semaphore.shared.inner) {
                    bail!("device-group semaphore belongs to a different Vulkan logical device");
                }
                if semaphore.is_timeline() && value == 0 {
                    bail!("device-group timeline semaphore values must be positive");
                }
            }
        }
        for wait in submission_waits {
            if !Arc::ptr_eq(&self.inner, &wait.inner) {
                bail!("submission timeline wait belongs to a different Vulkan logical device");
            }
            if wait.value == 0 {
                bail!("submission timeline semaphore values must be positive");
            }
        }

        self.inner
            .with_command_pool_host_access(|| unsafe {
                self.inner.device.end_command_buffer(self.command_buffer)
            })?
            .map_err(|err| anyhow!("ending Vulkan compute batch: {err:?}"))?;
        let completion = {
            let queue_index = self.device.queue_index;
            let queue = self.inner.queues[queue_index];
            let _queue_guard = self.inner.queue_locks[queue_index]
                .lock()
                .map_err(|_| anyhow!("Vulkan queue lock was poisoned"))?;
            let completion_timeline = self.inner.reserve_submission_timeline(queue_index)?;
            let fence = if completion_timeline.is_some() {
                vk::Fence::null()
            } else {
                unsafe {
                    self.inner
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                }
                .map_err(|err| anyhow!("creating Vulkan compute-batch fence: {err:?}"))?
            };
            let command_buffers = [self.command_buffer];
            let command_buffer_masks = [self.device.device_mask];
            let mut wait_semaphores = waits
                .iter()
                .map(|(semaphore, _)| semaphore.shared.semaphore)
                .collect::<Vec<_>>();
            wait_semaphores.extend(submission_waits.iter().map(|wait| wait.semaphore));
            let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_semaphores.len()];
            let mut signal_semaphores = signals
                .iter()
                .map(|(semaphore, _)| semaphore.shared.semaphore)
                .collect::<Vec<_>>();
            let mut wait_values = waits.iter().map(|(_, value)| *value).collect::<Vec<_>>();
            wait_values.extend(submission_waits.iter().map(|wait| wait.value));
            let mut signal_values = signals.iter().map(|(_, value)| *value).collect::<Vec<_>>();
            let wait_device_indices =
                vec![self.device.device_group_local_index; wait_semaphores.len()];
            let mut signal_device_indices =
                vec![self.device.device_group_local_index; signal_semaphores.len()];
            if let Some((semaphore, value)) = completion_timeline {
                signal_semaphores.push(semaphore);
                signal_values.push(value);
                signal_device_indices.push(self.device.device_group_local_index);
            }
            let uses_timeline = completion_timeline.is_some()
                || !submission_waits.is_empty()
                || waits.iter().any(|(semaphore, _)| semaphore.is_timeline())
                || signals.iter().any(|(semaphore, _)| semaphore.is_timeline());
            let mut group_submit = vk::DeviceGroupSubmitInfo::default()
                .wait_semaphore_device_indices(&wait_device_indices)
                .command_buffer_device_masks(&command_buffer_masks)
                .signal_semaphore_device_indices(&signal_device_indices);
            let mut timeline_submit = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let mut submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            if uses_timeline {
                submit = submit.push_next(&mut timeline_submit);
            }
            if self.device.is_multi_physical_device_logical_device() {
                submit = submit.push_next(&mut group_submit);
            }
            let submits = [submit];
            if let Err(err) = unsafe { self.inner.device.queue_submit(queue, &submits, fence) } {
                if fence != vk::Fence::null() {
                    unsafe { self.inner.device.destroy_fence(fence, None) };
                }
                return Err(anyhow!("submitting Vulkan compute batch: {err:?}"));
            }
            match completion_timeline {
                Some((semaphore, value)) => SubmissionCompletion::Timeline { semaphore, value },
                None => SubmissionCompletion::Fence(fence),
            }
        };
        self.into_submitted(completion)
    }

    fn into_submitted(mut self, completion: SubmissionCompletion) -> Result<SubmittedComputeBatch> {
        self.finished = true;
        let inner = Arc::clone(&self.inner);
        let batch = match completion {
            SubmissionCompletion::Timeline { semaphore, value } => {
                self.emit_dispatch_dependency_trace();
                self.retire_recyclable_buffers_on_timeline(semaphore, value);
                self.detach_timeline_submission_resources(semaphore, value)?;
                self.uses_persistent_upload_arena().then_some(self)
            }
            SubmissionCompletion::Fence(_) => Some(self),
        };
        Ok(SubmittedComputeBatch {
            inner,
            batch,
            completion,
            completed: false,
        })
    }

    fn finish_submission_diagnostics(&mut self) -> Result<()> {
        self.emit_kernel_timestamp_profile()?;
        self.emit_dispatch_dependency_trace();
        Ok(())
    }

    fn emit_dispatch_dependency_trace(&mut self) {
        if let Some(trace) = self.dispatch_dependency_trace.take() {
            let mut edges = trace.edges.iter().collect::<Vec<_>>();
            edges.sort_unstable_by(|left, right| right.1.cmp(left.1));
            for (&(producer, consumer, hazard), &count) in edges.into_iter().take(24) {
                let producer_name = shader_debug_name(producer).unwrap_or("unknown");
                let consumer_name = shader_debug_name(consumer).unwrap_or("unknown");
                eprintln!(
                    "hierarchos_vulkan_dependency_edge producer=0x{producer:016x} consumer=0x{consumer:016x} count={count} hazard={} producer_name={producer_name} consumer_name={consumer_name}",
                    hazard.as_str(),
                );
            }
        }
    }

    fn record_kernel_timestamp_begin(&mut self, shader_signature: u64) -> Result<()> {
        let Some(profile) = self.kernel_timestamp_profile.as_mut() else {
            return Ok(());
        };
        let needs_chunk = profile.chunks.last().is_none_or(|chunk| {
            chunk.shader_signatures.len() == KERNEL_PROFILE_DISPATCHES_PER_POOL
        });
        if needs_chunk {
            let query_count = u32::try_from(KERNEL_PROFILE_DISPATCHES_PER_POOL * 2)
                .context("Vulkan kernel-profile query count exceeds u32 range")?;
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(query_count);
            let pool = unsafe { self.inner.device.create_query_pool(&info, None) }
                .map_err(|err| anyhow!("creating Vulkan kernel-profile timestamp pool: {err:?}"))?;
            self.inner.with_command_pool_host_access(|| unsafe {
                self.inner
                    .device
                    .cmd_reset_query_pool(self.command_buffer, pool, 0, query_count);
            })?;
            profile.chunks.push(KernelTimestampQueryChunk {
                pool,
                shader_signatures: Vec::with_capacity(KERNEL_PROFILE_DISPATCHES_PER_POOL),
            });
        }
        let chunk = profile
            .chunks
            .last_mut()
            .context("Vulkan kernel profiler lost its active query pool")?;
        let query = u32::try_from(chunk.shader_signatures.len() * 2)
            .context("Vulkan kernel-profile query index exceeds u32 range")?;
        self.inner.with_command_pool_host_access(|| unsafe {
            self.inner.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                chunk.pool,
                query,
            );
        })?;
        chunk.shader_signatures.push(shader_signature);
        Ok(())
    }

    fn record_kernel_timestamp_end(&mut self) -> Result<()> {
        let Some(profile) = self.kernel_timestamp_profile.as_mut() else {
            return Ok(());
        };
        let chunk = profile
            .chunks
            .last()
            .context("Vulkan kernel profiler has no timestamp pool for dispatch end")?;
        let query = u32::try_from(chunk.shader_signatures.len() * 2 - 1)
            .context("Vulkan kernel-profile query index exceeds u32 range")?;
        self.inner.with_command_pool_host_access(|| unsafe {
            self.inner.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                chunk.pool,
                query,
            );
        })?;
        Ok(())
    }

    fn emit_kernel_timestamp_profile(&self) -> Result<()> {
        let Some(profile) = self.kernel_timestamp_profile.as_ref() else {
            return Ok(());
        };
        emit_kernel_timestamp_profile_for_inner(&self.inner, profile)
    }

    fn shader_barrier(&mut self) -> Result<()> {
        let barriers = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        self.inner.with_command_pool_host_access(|| unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
                &[],
            );
        })?;
        self.shader_barrier_count = self
            .shader_barrier_count
            .checked_add(1)
            .context("Vulkan compute-batch shader-barrier count overflow")?;
        Ok(())
    }

    /// Insert a compute-to-compute memory barrier only when the next dispatch
    /// can alias a buffer used by any dispatch recorded since the last shader
    /// barrier.
    ///
    /// Kernels default every storage binding to conservative read/write access;
    /// explicitly audited GLSL bindings can narrow that to read-only or
    /// write-only. This keeps the elision rule safe without shader reflection
    /// while allowing proven read/read overlaps to execute without a barrier.
    fn prepare_shader_dispatch(
        &mut self,
        shader_signature: u64,
        buffers: &[&GpuBuffer],
        binding_accesses: &[BindingAccess],
    ) -> Result<()> {
        let overlaps_pending = buffers
            .iter()
            .zip(binding_accesses)
            .any(|(buffer, access)| {
                let region = buffer.buffer_region_key();
                (access.may_read() && self.pending_shader_writes.contains(&region))
                    || (access.may_write() && self.pending_shader_buffers.contains(&region))
            });
        if overlaps_pending {
            if let Some(trace) = self.dispatch_dependency_trace.as_mut() {
                for (buffer, access) in buffers.iter().zip(binding_accesses) {
                    let region = buffer.buffer_region_key();
                    if access.may_read() && self.pending_shader_writes.contains(&region) {
                        if let Some(&producer) = trace.pending_shader_write_owners.get(&region) {
                            let edge = trace
                                .edges
                                .entry((producer, shader_signature, DispatchHazard::ReadAfterWrite))
                                .or_default();
                            *edge = edge.saturating_add(1);
                        }
                    }
                    if access.may_write() && self.pending_shader_reads.contains(&region) {
                        if let Some(&producer) = trace.pending_shader_read_owners.get(&region) {
                            let edge = trace
                                .edges
                                .entry((producer, shader_signature, DispatchHazard::WriteAfterRead))
                                .or_default();
                            *edge = edge.saturating_add(1);
                        }
                    }
                    if access.may_write() && self.pending_shader_writes.contains(&region) {
                        if let Some(&producer) = trace.pending_shader_write_owners.get(&region) {
                            let edge = trace
                                .edges
                                .entry((
                                    producer,
                                    shader_signature,
                                    DispatchHazard::WriteAfterWrite,
                                ))
                                .or_default();
                            *edge = edge.saturating_add(1);
                        }
                    }
                }
            }
            self.shader_barrier()?;
            self.pending_shader_buffers.clear();
            self.pending_shader_reads.clear();
            self.pending_shader_writes.clear();
            if let Some(trace) = self.dispatch_dependency_trace.as_mut() {
                trace.pending_shader_read_owners.clear();
                trace.pending_shader_write_owners.clear();
            }
        }
        Ok(())
    }

    fn finish_shader_dispatch(
        &mut self,
        shader_signature: u64,
        buffers: &[&GpuBuffer],
        binding_accesses: &[BindingAccess],
    ) {
        for (buffer, access) in buffers.iter().zip(binding_accesses) {
            let region = buffer.buffer_region_key();
            self.pending_shader_buffers.insert(region);
            if access.may_read() {
                self.pending_shader_reads.insert(region);
            }
            if access.may_write() {
                self.pending_shader_writes.insert(region);
            }
        }
        if let Some(trace) = self.dispatch_dependency_trace.as_mut() {
            for (buffer, access) in buffers.iter().zip(binding_accesses) {
                let region = buffer.buffer_region_key();
                if access.may_read() {
                    trace
                        .pending_shader_read_owners
                        .insert(region, shader_signature);
                }
                if access.may_write() {
                    trace
                        .pending_shader_write_owners
                        .insert(region, shader_signature);
                }
            }
        }
    }
}

impl SubmittedComputeBatch {
    pub(crate) fn timeline_wait(&self) -> Option<SubmissionTimelineWait> {
        match self.completion {
            SubmissionCompletion::Timeline { semaphore, value } => Some(SubmissionTimelineWait {
                inner: Arc::clone(&self.inner),
                semaphore,
                value,
            }),
            SubmissionCompletion::Fence(_) => None,
        }
    }

    pub(crate) fn wait(mut self) -> Result<()> {
        self.wait_inner()
    }

    fn wait_inner(&mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        match self.completion {
            SubmissionCompletion::Fence(fence) => unsafe {
                let batch = self
                    .batch
                    .as_mut()
                    .context("fence-backed Vulkan submission lost its batch resource owner")?;
                batch
                    .inner
                    .device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|err| anyhow!("waiting for Vulkan compute-batch fence: {err:?}"))?;
                batch.inner.device.destroy_fence(fence, None);
                self.completed = true;
                return batch.finish_submission_diagnostics();
            },
            SubmissionCompletion::Timeline { semaphore, value } => {
                self.inner.wait_submission_timeline(semaphore, value)?;
                let _ = self.inner.reap_completed_submission_resources()?;
            }
        }
        self.completed = true;
        if let Some(batch) = self.batch.as_mut() {
            batch.finish_submission_diagnostics()?;
        }
        Ok(())
    }
}

impl SubmissionTimelineWait {
    pub(crate) fn is_complete(&self) -> Result<bool> {
        let observed = self
            .inner
            .submission_timeline_counter_value(self.semaphore)?;
        let complete = observed >= self.value;
        if complete {
            let _ = self.inner.reap_completed_submission_resources()?;
        }
        Ok(complete)
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.inner
            .wait_submission_timeline(self.semaphore, self.value)?;
        let _ = self.inner.reap_completed_submission_resources()?;
        Ok(())
    }
}

impl Drop for SubmittedComputeBatch {
    fn drop(&mut self) {
        if !self.completed {
            // Resource safety is more important than surfacing a secondary
            // cleanup error: the explicit `wait()` path reports failures. On
            // timeline devices ordinary transient resources are owned by the
            // device arena, so dropping a lightweight submission handle does
            // not join the queue. Persistent upload arenas are the exception:
            // their staging chunks must not be reused before completion.
            match self.completion {
                SubmissionCompletion::Fence(fence) => {
                    if let Some(batch) = self.batch.as_ref() {
                        unsafe {
                            let _ = batch.inner.device.wait_for_fences(&[fence], true, u64::MAX);
                            batch.inner.device.destroy_fence(fence, None);
                        }
                    }
                }
                SubmissionCompletion::Timeline { semaphore, value } => {
                    if self.batch.is_some() {
                        let _ = self.inner.wait_submission_timeline(semaphore, value);
                    }
                    let _ = self.inner.reap_completed_submission_resources();
                }
            }
        }
    }
}

impl Drop for ComputeBatch {
    fn drop(&mut self) {
        unsafe {
            if let Some(profile) = self.kernel_timestamp_profile.as_mut() {
                for chunk in profile.chunks.drain(..) {
                    self.inner.device.destroy_query_pool(chunk.pool, None);
                }
            }
        }
        for chunk in self.descriptor_pools.drain(..) {
            self.inner.recycle_descriptor_pool_chunk(chunk);
        }
        self.inner
            .recycle_compute_command_buffer(self.command_buffer);
        self.command_buffer = vk::CommandBuffer::null();
        if let UploadArenaStorage::Persistent(arena) = &self.upload_arena {
            arena.release();
        }
    }
}

fn emit_kernel_timestamp_profile_for_inner(
    inner: &DeviceInner,
    profile: &KernelTimestampProfile,
) -> Result<()> {
    let mut by_shader = HashMap::<u64, KernelTimestampStats>::new();
    let mut total_ticks = 0u128;
    let mut total_dispatches = 0usize;
    for chunk in &profile.chunks {
        if chunk.shader_signatures.is_empty() {
            continue;
        }
        let query_count = chunk
            .shader_signatures
            .len()
            .checked_mul(2)
            .context("Vulkan kernel-profile result count overflow")?;
        let mut timestamps = vec![0u64; query_count];
        unsafe {
            inner
                .device
                .get_query_pool_results(
                    chunk.pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|err| anyhow!("reading Vulkan kernel-profile timestamps: {err:?}"))?;
        }
        for (index, &shader_signature) in chunk.shader_signatures.iter().enumerate() {
            let start = timestamps[index * 2];
            let end = timestamps[index * 2 + 1];
            let ticks = timestamp_delta(start, end, profile.timestamp_valid_bits) as u128;
            let stats = by_shader.entry(shader_signature).or_default();
            stats.dispatches = stats.dispatches.saturating_add(1);
            stats.ticks = stats.ticks.saturating_add(ticks);
            total_ticks = total_ticks.saturating_add(ticks);
            total_dispatches = total_dispatches.saturating_add(1);
        }
    }

    let total_gpu_ms = total_ticks as f64 * profile.timestamp_period_ns / 1_000_000.0;
    let total_gpu_ns = (total_ticks as f64 * profile.timestamp_period_ns)
        .round()
        .clamp(0.0, u64::MAX as f64) as u64;
    inner
        .kernel_timestamp_profile_samples
        .fetch_add(1, Ordering::AcqRel);
    inner.kernel_timestamp_profile_dispatches.fetch_add(
        u64::try_from(total_dispatches).unwrap_or(u64::MAX),
        Ordering::AcqRel,
    );
    inner
        .kernel_timestamp_profile_gpu_ns_total
        .fetch_add(total_gpu_ns, Ordering::AcqRel);
    if !profile.report_diagnostics {
        return Ok(());
    }
    eprintln!(
        "hierarchos_vulkan_kernel_profile device={:?} dispatches={} gpu_ms={:.6} timestamp_period_ns={:.6}",
        inner.name,
        total_dispatches,
        total_gpu_ms,
        profile.timestamp_period_ns
    );

    let mut by_category = HashMap::<&'static str, KernelTimestampStats>::new();
    for (&signature, stats) in &by_shader {
        let name = shader_debug_name(signature).unwrap_or("unknown");
        let category = kernel_profile_category(name);
        let aggregate = by_category.entry(category).or_default();
        aggregate.dispatches = aggregate.dispatches.saturating_add(stats.dispatches);
        aggregate.ticks = aggregate.ticks.saturating_add(stats.ticks);
    }
    let mut categories = by_category.into_iter().collect::<Vec<_>>();
    categories.sort_unstable_by(|left, right| right.1.ticks.cmp(&left.1.ticks));
    for (category, stats) in categories {
        let gpu_ms = stats.ticks as f64 * profile.timestamp_period_ns / 1_000_000.0;
        let pct = if total_ticks == 0 {
            0.0
        } else {
            stats.ticks as f64 * 100.0 / total_ticks as f64
        };
        eprintln!(
            "hierarchos_vulkan_kernel_profile_category category={} dispatches={} gpu_ms={:.6} pct={:.3}",
            category, stats.dispatches, gpu_ms, pct
        );
    }

    let mut shaders = by_shader.into_iter().collect::<Vec<_>>();
    shaders.sort_unstable_by(|left, right| right.1.ticks.cmp(&left.1.ticks));
    for (signature, stats) in shaders.into_iter().take(KERNEL_PROFILE_TOP_SHADERS) {
        let name = shader_debug_name(signature).unwrap_or("unknown");
        let gpu_ms = stats.ticks as f64 * profile.timestamp_period_ns / 1_000_000.0;
        let pct = if total_ticks == 0 {
            0.0
        } else {
            stats.ticks as f64 * 100.0 / total_ticks as f64
        };
        eprintln!(
            "hierarchos_vulkan_kernel_profile_shader shader={} category={} dispatches={} gpu_ms={:.6} pct={:.3}",
            name,
            kernel_profile_category(name),
            stats.dispatches,
            gpu_ms,
            pct
        );
    }
    Ok(())
}

fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> u64 {
    if valid_bits >= 64 {
        return end.wrapping_sub(start);
    }
    let modulus = 1u64 << valid_bits;
    let mask = modulus - 1;
    end.wrapping_sub(start) & mask
}

fn kernel_profile_category(shader_name: &str) -> &'static str {
    if shader_name == "adamw_fp16_mirror" {
        "optimizer.adamw-fp16-mirror"
    } else if shader_name == "adamw" {
        "optimizer.adamw"
    } else if matches!(shader_name, "fp32_to_fp16_packed" | "fp32_to_bf16_packed") {
        "mixed-precision.mirror-pack"
    } else if matches!(shader_name, "fp16_packed_to_fp32" | "bf16_packed_to_fp32") {
        "mixed-precision.mirror-unpack"
    } else if matches!(
        shader_name,
        "gradient_accumulate" | "gradient_accumulate4" | "gradient_scale"
    ) {
        "optimizer.gradient-bookkeeping"
    } else if shader_name.starts_with("cross_entropy_linear_") {
        "loss.lm-projection"
    } else if shader_name.starts_with("cross_entropy_") {
        "loss.cross-entropy"
    } else if shader_name.starts_with("embedding_") {
        "token.embedding"
    } else if shader_name.starts_with("rwkv_") || shader_name.starts_with("packed_cell_") {
        "rwkv.recurrent"
    } else if shader_name.starts_with("layer_norm_") {
        "normalization.layer-norm"
    } else if shader_name.starts_with("linear") || shader_name.starts_with("parameter_matmul_") {
        "projection.linear"
    } else if shader_name.starts_with("rosa_") {
        "memory.rosa"
    } else if shader_name.starts_with("hard_act_")
        || shader_name.starts_with("context_lerp_")
        || shader_name.starts_with("drift_update_")
        || shader_name.starts_with("row_keep_")
        || shader_name.starts_with("worker_convergence")
        || shader_name.starts_with("commitment_")
        || shader_name.starts_with("indexed_step_")
    {
        "control"
    } else if shader_name.starts_with("bias_")
        || shader_name.starts_with("channel_reduce")
        || shader_name.starts_with("finite_clamp_")
        || shader_name.starts_with("gelu_")
        || shader_name.starts_with("relu2_")
        || shader_name.starts_with("sigmoid_")
        || shader_name.starts_with("silu_")
        || shader_name.starts_with("tanh_")
        || shader_name.starts_with("vector_add")
    {
        "elementwise-reduction"
    } else {
        "other"
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
}

pub(crate) struct ComputeKernel {
    inner: Arc<DeviceInner>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_layout_signature: DescriptorLayoutSignature,
    pipeline_layout_signature: PipelineLayoutSignature,
    shader_signature: u64,
    binding_accesses: Vec<BindingAccess>,
    pipeline: vk::Pipeline,
    binding_count: usize,
    push_constant_size: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BindingAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    MayWrite,
}

impl BindingAccess {
    fn may_read(self) -> bool {
        !matches!(self, Self::WriteOnly)
    }

    fn may_write(self) -> bool {
        !matches!(self, Self::ReadOnly)
    }
}

impl ComputeKernel {
    #[cfg(test)]
    pub(crate) fn shares_interned_layouts_with(&self, other: &Self) -> bool {
        self.descriptor_set_layout == other.descriptor_set_layout
            && self.pipeline_layout == other.pipeline_layout
            && self.descriptor_layout_signature == other.descriptor_layout_signature
            && self.pipeline_layout_signature == other.pipeline_layout_signature
    }

    pub(crate) fn new(
        device: &VulkanDevice,
        spirv: &[u8],
        binding_count: usize,
        push_constant_size: u32,
    ) -> Result<Self> {
        Self::new_with_access(
            device,
            spirv,
            &vec![BindingAccess::MayWrite; binding_count],
            push_constant_size,
        )
    }

    pub(crate) fn new_with_access(
        device: &VulkanDevice,
        spirv: &[u8],
        binding_accesses: &[BindingAccess],
        push_constant_size: u32,
    ) -> Result<Self> {
        let binding_count = binding_accesses.len();
        if binding_count == 0 {
            bail!("compute kernel requires at least one storage buffer binding");
        }
        let binding_count_u32 = u32::try_from(binding_count)
            .context("compute kernel binding count exceeds Vulkan u32 range")?;
        let descriptor_layout_signature = DescriptorLayoutSignature {
            binding_count: binding_count_u32,
        };
        let pipeline_layout_signature = PipelineLayoutSignature {
            descriptor: descriptor_layout_signature,
            push_constant_size,
        };
        let shader_signature = fnv1a64(spirv);
        let (descriptor_set_layout, pipeline_layout) = intern_kernel_layouts(
            device,
            descriptor_layout_signature,
            pipeline_layout_signature,
        )?;

        let words = read_spv(&mut Cursor::new(spirv)).context("decoding embedded SPIR-V")?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let shader_module =
            match unsafe { device.inner.device.create_shader_module(&shader_info, None) } {
                Ok(module) => module,
                Err(err) => return Err(anyhow!("creating Vulkan shader module: {err:?}")),
            };
        let entry_name = CString::new("main")?;
        let mut required_subgroup_size_info =
            vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(device.inner.required_subgroup_size.unwrap_or_default());
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);
        if device.inner.required_subgroup_size.is_some() {
            stage = stage.push_next(&mut required_subgroup_size_info);
        }
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];
        let pipeline_result = unsafe {
            device.inner.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &pipeline_info,
                None,
            )
        };
        unsafe {
            device
                .inner
                .device
                .destroy_shader_module(shader_module, None)
        };
        let pipeline = match pipeline_result {
            Ok(pipelines) => pipelines[0],
            Err((_, err)) => return Err(anyhow!("creating Vulkan compute pipeline: {err:?}")),
        };

        Ok(Self {
            inner: Arc::clone(&device.inner),
            descriptor_set_layout,
            pipeline_layout,
            descriptor_layout_signature,
            pipeline_layout_signature,
            shader_signature,
            binding_accesses: binding_accesses.to_vec(),
            pipeline,
            binding_count,
            push_constant_size,
        })
    }

    pub(crate) fn record_dispatch(
        &self,
        batch: &mut ComputeBatch,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &batch.inner) {
            bail!("compute kernel and batch belong to different Vulkan devices");
        }
        if buffers.len() != self.binding_count {
            bail!(
                "kernel expected {} buffers, got {}",
                self.binding_count,
                buffers.len()
            );
        }
        if push_constants.len() != self.push_constant_size as usize {
            bail!(
                "kernel expected {} push-constant bytes, got {}",
                self.push_constant_size,
                push_constants.len()
            );
        }
        if groups.contains(&0) {
            bail!("Vulkan dispatch group counts must be positive");
        }

        batch.retain_recyclable_buffers(buffers);

        batch.prepare_shader_dispatch(self.shader_signature, buffers, &self.binding_accesses)?;

        batch.dispatch_count = batch
            .dispatch_count
            .checked_add(1)
            .context("Vulkan compute-batch dispatch count overflow")?;

        if batch.bound_pipeline != Some(self.pipeline) {
            self.inner.with_command_pool_host_access(|| unsafe {
                self.inner.device.cmd_bind_pipeline(
                    batch.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline,
                );
            })?;
            batch.bound_pipeline = Some(self.pipeline);
            batch.pipeline_bind_count = batch
                .pipeline_bind_count
                .checked_add(1)
                .context("Vulkan compute-batch pipeline-bind count overflow")?;
        }

        let descriptor_is_bound = batch.bound_descriptor_pipeline_layout
            == Some(self.pipeline_layout_signature)
            && batch.bound_descriptor_layout == Some(self.descriptor_layout_signature)
            && batch.bound_descriptor_set.is_some()
            && batch.bound_descriptor_buffers.len() == buffers.len()
            && batch
                .bound_descriptor_buffers
                .iter()
                .zip(buffers)
                .all(|(bound, buffer)| *bound == buffer.buffer_region_key());
        if !descriptor_is_bound {
            let descriptor_set = batch.descriptor_set_for(
                self.descriptor_set_layout,
                self.descriptor_layout_signature,
                self.binding_count,
                buffers,
            )?;
            self.inner.with_command_pool_host_access(|| unsafe {
                self.inner.device.cmd_bind_descriptor_sets(
                    batch.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
            })?;
            batch.bound_descriptor_pipeline_layout = Some(self.pipeline_layout_signature);
            batch.bound_descriptor_layout = Some(self.descriptor_layout_signature);
            batch.bound_descriptor_set = Some(descriptor_set);
            batch.bound_descriptor_buffers.clear();
            batch
                .bound_descriptor_buffers
                .extend(buffers.iter().map(|buffer| buffer.buffer_region_key()));
            batch.descriptor_bind_count = batch
                .descriptor_bind_count
                .checked_add(1)
                .context("Vulkan compute-batch descriptor-bind count overflow")?;
        }

        let push_constants_are_current = batch.pushed_constant_layout
            == Some(self.pipeline_layout_signature)
            && batch.pushed_constants.as_slice() == push_constants;
        if !push_constants_are_current {
            self.inner.with_command_pool_host_access(|| unsafe {
                self.inner.device.cmd_push_constants(
                    batch.command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            })?;
            batch.pushed_constant_layout = Some(self.pipeline_layout_signature);
            batch.pushed_constants.clear();
            batch.pushed_constants.extend_from_slice(push_constants);
            batch.push_constant_write_count = batch
                .push_constant_write_count
                .checked_add(1)
                .context("Vulkan compute-batch push-constant write count overflow")?;
        }

        batch.record_kernel_timestamp_begin(self.shader_signature)?;
        self.inner.with_command_pool_host_access(|| unsafe {
            self.inner
                .device
                .cmd_dispatch(batch.command_buffer, groups[0], groups[1], groups[2]);
        })?;
        batch.record_kernel_timestamp_end()?;
        batch.finish_shader_dispatch(self.shader_signature, buffers, &self.binding_accesses);
        Ok(())
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl Drop for ComputeKernel {
    fn drop(&mut self) {
        unsafe {
            self.inner.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        insert_free_range, intern_kernel_layouts, kernel_profile_category, take_aligned_range,
        timestamp_delta, ComputeBatch, ComputeKernel, DescriptorLayoutSignature, GpuBuffer,
        MemoryRange, PipelineLayoutSignature, SubmissionCompletion, UploadArenaStorage,
        VulkanDevice, VulkanDeviceGroupInfo, VulkanExternalBufferCapabilities,
        VulkanExternalSemaphoreCapabilities, VulkanGradientTransportBackend, VulkanMemoryBudget,
        VulkanPhysicalDeviceInfo,
    };
    use anyhow::{anyhow, Context, Result};

    fn external_memory_test_device(
        index: usize,
        device_uuid: &str,
        driver_uuid: &str,
        group_index: usize,
    ) -> VulkanPhysicalDeviceInfo {
        VulkanPhysicalDeviceInfo {
            index,
            name: format!("test-gpu-{index}"),
            device_type: "DISCRETE_GPU".to_string(),
            compute_queue_family_index: 0,
            device_uuid: device_uuid.to_string(),
            driver_uuid: driver_uuid.to_string(),
            device_group: Some(VulkanDeviceGroupInfo {
                group_index,
                physical_device_count: 2,
                subset_allocation: true,
            }),
            external_buffer: VulkanExternalBufferCapabilities {
                opaque_win32_extension_exposed: true,
                opaque_win32_exportable: true,
                opaque_win32_importable: true,
                opaque_fd_extension_exposed: true,
                opaque_fd_exportable: true,
                opaque_fd_importable: true,
            },
            external_semaphore: VulkanExternalSemaphoreCapabilities {
                opaque_win32_extension_exposed: true,
                opaque_win32_exportable: true,
                opaque_win32_importable: true,
                opaque_fd_extension_exposed: true,
                opaque_fd_exportable: true,
                opaque_fd_importable: true,
            },
        }
    }

    #[test]
    fn timestamp_delta_handles_wrapped_queue_timestamps() {
        assert_eq!(timestamp_delta(12, 29, 64), 17);
        assert_eq!(timestamp_delta(250, 7, 8), 13);
    }

    #[test]
    fn submitted_compute_batch_keeps_copy_resources_alive_until_completion_wait() -> Result<()> {
        let device = VulkanDevice::new()?;
        let source = GpuBuffer::from_f32(&device, &[1.25, -2.5, 3.75, 4.5])?;
        let destination = GpuBuffer::zeros_f32(&device, 4)?;
        let mut commands = ComputeBatch::new(&device)?;
        commands.copy_f32_range(&source, 0, &destination, 0, 4)?;
        let submitted = commands.submit_async()?;
        submitted.wait()?;
        assert_eq!(destination.read_f32(4)?, vec![1.25, -2.5, 3.75, 4.5]);
        Ok(())
    }

    #[test]
    fn completed_compute_batches_reuse_command_buffers() -> Result<()> {
        let device = VulkanDevice::new()?;
        let initial = {
            let ring = device
                .inner
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            (ring.allocated, ring.reused)
        };

        let first = ComputeBatch::new(&device)?.submit_async()?;
        if device.inner.submission_timeline_enabled {
            assert!(matches!(
                first.completion,
                SubmissionCompletion::Timeline { .. }
            ));
        }
        first.wait()?;
        let after_first = {
            let ring = device
                .inner
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            (ring.allocated, ring.reused, ring.reusable.len())
        };
        assert_eq!(after_first.0, initial.0.saturating_add(1));
        assert_eq!(after_first.1, initial.1);
        assert_eq!(after_first.2, 1);

        ComputeBatch::new(&device)?.submit()?;
        let after_second = {
            let ring = device
                .inner
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            (ring.allocated, ring.reused, ring.reusable.len())
        };
        assert_eq!(after_second.0, after_first.0);
        assert_eq!(after_second.1, after_first.1.saturating_add(1));
        assert_eq!(after_second.2, 1);
        Ok(())
    }

    #[test]
    fn timeline_command_buffer_ring_reclaims_before_submission_owner_drop() -> Result<()> {
        let device = VulkanDevice::new()?;
        if !device.inner.submission_timeline_enabled {
            eprintln!(
                "skipping timeline command-buffer retirement test: timeline semaphore unavailable"
            );
            return Ok(());
        }

        let initial = {
            let ring = device
                .inner
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            (ring.allocated, ring.timeline_reaped)
        };
        let submitted = ComputeBatch::new(&device)?.submit_async()?;
        let completion = submitted
            .timeline_wait()
            .context("timeline submission did not expose a detached completion wait")?;
        {
            let arena = device
                .inner
                .submission_resource_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
            assert_eq!(arena.in_flight.len(), 1);
        }
        {
            let ring = device
                .inner
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            assert_eq!(ring.reusable.len(), 0);
        }

        while !completion.is_complete()? {
            std::thread::yield_now();
        }
        let reclaimed = {
            let ring = device
                .inner
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            (ring.allocated, ring.timeline_reaped, ring.reusable.len())
        };
        let arena_in_flight = device
            .inner
            .submission_resource_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?
            .in_flight
            .len();
        assert_eq!(reclaimed.0, initial.0.saturating_add(1));
        assert_eq!(reclaimed.1, initial.1.saturating_add(1));
        assert_eq!(arena_in_flight, 0);
        assert_eq!(reclaimed.2, 1);

        // Keep `submitted` alive while a second batch checks the recycled raw
        // command buffer back out. Timeline-owned transient resources are
        // already retired independently of the lightweight submission handle.
        let second = ComputeBatch::new(&device)?.submit_async()?;
        let after_second_submit = {
            let ring = device
                .inner
                .command_buffer_ring
                .lock()
                .map_err(|_| anyhow!("Vulkan command-buffer ring lock was poisoned"))?;
            (ring.allocated, ring.reused)
        };
        assert_eq!(after_second_submit.0, reclaimed.0);
        assert!(after_second_submit.1 >= 1);
        second.wait()?;
        submitted.wait()?;
        Ok(())
    }

    #[test]
    fn device_fill_f32_writes_pattern_without_upload_staging() -> Result<()> {
        let device = VulkanDevice::new()?;
        let destination = GpuBuffer::zeros_f32(&device, 4)?;
        let mut commands = ComputeBatch::new(&device)?;

        commands.fill_f32(&destination, 4, 1.0)?;
        assert_eq!(commands.uploaded_bytes(), 0);
        assert_eq!(commands.upload_arena_buffer_count(), 0);
        commands.submit()?;

        assert_eq!(destination.read_f32(4)?, vec![1.0; 4]);
        Ok(())
    }

    #[test]
    fn device_update_f32_at_writes_scalar_without_upload_staging() -> Result<()> {
        let device = VulkanDevice::new()?;
        let destination = GpuBuffer::zeros_f32(&device, 4)?;
        let mut commands = ComputeBatch::new(&device)?;

        commands.fill_zero_f32(&destination, 4)?;
        commands.update_f32_at(&destination, 2, 3.5)?;
        assert_eq!(commands.uploaded_bytes(), 0);
        assert_eq!(commands.upload_arena_buffer_count(), 0);
        commands.submit()?;

        assert_eq!(destination.read_f32(4)?, vec![0.0, 0.0, 3.5, 0.0]);
        Ok(())
    }

    #[test]
    fn timeline_submission_arena_detaches_local_upload_chunks_from_owner() -> Result<()> {
        let device = VulkanDevice::new()?;
        if !device.inner.submission_timeline_enabled {
            eprintln!(
                "skipping timeline submission-resource retirement test: timeline semaphore unavailable"
            );
            return Ok(());
        }

        let destination = GpuBuffer::zeros_f32(&device, 4)?;
        let mut commands = ComputeBatch::new(&device)?;
        commands.upload_f32(&destination, &[1.0, 2.0, 3.0, 4.0])?;
        assert_eq!(commands.upload_arena_buffer_count(), 1);
        let submitted = commands.submit_async()?;
        assert!(submitted.batch.is_none());
        let completion = submitted
            .timeline_wait()
            .context("timeline submission did not expose a detached completion wait")?;
        {
            let arena = device
                .inner
                .submission_resource_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
            assert_eq!(arena.in_flight.len(), 1);
            assert_eq!(arena.in_flight[0].local_upload_allocations.len(), 1);
        }
        // The higher-level owner can disappear immediately. The standalone
        // timeline dependency is sufficient to protect and later reclaim every
        // transient resource detached into the device arena.
        drop(submitted);
        completion.wait()?;
        assert_eq!(destination.read_f32(4)?, vec![1.0, 2.0, 3.0, 4.0]);
        let (recycled_buffer, reuse_before) = {
            let arena = device
                .inner
                .submission_resource_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
            assert!(arena.in_flight.is_empty());
            assert_eq!(arena.reusable_buffer_allocations.len(), 1);
            (
                arena.reusable_buffer_allocations[0].buffer,
                arena.buffer_allocation_reused,
            )
        };

        let mut reused = ComputeBatch::new(&device)?;
        reused.upload_f32(&destination, &[4.0, 3.0, 2.0, 1.0])?;
        match &reused.upload_arena {
            UploadArenaStorage::Local(chunks) => {
                assert_eq!(chunks.len(), 1);
                assert_eq!(chunks[0].buffer.allocation.buffer, recycled_buffer);
            }
            UploadArenaStorage::Persistent(_) => {
                panic!("plain compute batch unexpectedly used a persistent upload arena")
            }
        }
        let reuse_after = device
            .inner
            .submission_resource_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?
            .buffer_allocation_reused;
        assert_eq!(reuse_after, reuse_before.saturating_add(1));
        reused.submit()?;
        assert_eq!(destination.read_f32(4)?, vec![4.0, 3.0, 2.0, 1.0]);
        Ok(())
    }

    #[test]
    fn timeline_submission_arena_reuses_descriptor_pool_chunks() -> Result<()> {
        let device = VulkanDevice::new()?;
        if !device.inner.submission_timeline_enabled {
            eprintln!(
                "skipping timeline descriptor-pool reuse test: timeline semaphore unavailable"
            );
            return Ok(());
        }

        let descriptor_signature = DescriptorLayoutSignature { binding_count: 4 };
        let pipeline_signature = PipelineLayoutSignature {
            descriptor: descriptor_signature,
            push_constant_size: 0,
        };
        let (layout, _) = intern_kernel_layouts(&device, descriptor_signature, pipeline_signature)?;

        let mut first = ComputeBatch::new(&device)?;
        let _ = first.allocate_descriptor_set(layout, 4)?;
        let first_pool = first.descriptor_pools[0].pool;
        let submitted = first.submit_async()?;
        let completion = submitted
            .timeline_wait()
            .context("timeline descriptor submission did not expose a completion wait")?;
        drop(submitted);
        completion.wait()?;

        let latency_stats = device.submission_arena_stats()?;
        assert!(latency_stats.timeline_retirement_latency_samples >= 1);
        assert!(
            latency_stats.timeline_retirement_latency_ns_total
                >= latency_stats.timeline_retirement_latency_ns_max
        );
        assert!(
            latency_stats.timeline_retirement_latency_ns_average
                <= latency_stats.timeline_retirement_latency_ns_max
        );

        let reuse_before = {
            let arena = device
                .inner
                .submission_resource_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
            assert!(arena
                .reusable_descriptor_pools
                .iter()
                .any(|chunk| chunk.pool == first_pool));
            arena.descriptor_pool_reused
        };

        let mut second = ComputeBatch::new(&device)?;
        let _ = second.allocate_descriptor_set(layout, 4)?;
        assert_eq!(second.descriptor_pools[0].pool, first_pool);
        let reuse_after = device
            .inner
            .submission_resource_arena
            .lock()
            .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?
            .descriptor_pool_reused;
        assert_eq!(reuse_after, reuse_before.saturating_add(1));
        second.submit()?;
        Ok(())
    }

    #[test]
    fn transient_optimizer_scratch_suballocates_persistent_device_slab() -> Result<()> {
        let device = VulkanDevice::new()?;
        let first = GpuBuffer::transient_f32(&device, 1024)?;
        let first_buffer = first.allocation.buffer;
        let first_offset = first.allocation.buffer_offset_bytes;
        let second = GpuBuffer::transient_f32(&device, 1024)?;
        assert_eq!(second.allocation.buffer, first_buffer);
        assert_ne!(second.allocation.buffer_offset_bytes, first_offset);
        let stats = device.submission_arena_stats()?;
        assert_eq!(stats.scratch_slab_count, 1);
        assert_eq!(stats.scratch_live_leases, 2);
        assert!(stats.scratch_slab_capacity_bytes >= 2 * 1024 * 4);

        drop(first);
        let reused = GpuBuffer::transient_f32(&device, 1024)?;
        assert_eq!(reused.allocation.buffer, first_buffer);
        assert_eq!(reused.allocation.buffer_offset_bytes, first_offset);
        Ok(())
    }

    #[test]
    fn scratch_slab_nonzero_offsets_bind_distinct_descriptor_ranges() -> Result<()> {
        let device = VulkanDevice::new()?;
        let input = GpuBuffer::transient_f32(&device, 4)?;
        let weight = GpuBuffer::transient_f32(&device, 4)?;
        let output = GpuBuffer::transient_f32(&device, 1)?;
        assert_eq!(input.allocation.buffer, weight.allocation.buffer);
        assert_eq!(input.allocation.buffer, output.allocation.buffer);
        assert_ne!(
            input.allocation.buffer_offset_bytes,
            weight.allocation.buffer_offset_bytes
        );
        assert_ne!(
            weight.allocation.buffer_offset_bytes,
            output.allocation.buffer_offset_bytes
        );

        let kernel = ComputeKernel::new(
            &device,
            include_bytes!("../shaders/linear_forward.spv"),
            3,
            16,
        )?;
        let push_words = [1u32, 4, 1, 0];
        let mut commands = ComputeBatch::new(&device)?;
        commands.upload_f32(&input, &[1.0, 2.0, 3.0, 4.0])?;
        commands.upload_f32(&weight, &[2.0, 3.0, 4.0, 5.0])?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &weight, &output],
            bytemuck::cast_slice(&push_words),
            [1, 1, 1],
        )?;
        commands.submit()?;

        let actual = output.read_f32(1)?[0];
        assert!(
            (actual - 40.0).abs() <= 1.0e-6,
            "nonzero-offset slab dispatch produced {actual}"
        );
        Ok(())
    }

    #[test]
    fn timeline_submission_arena_defers_transient_scratch_reuse_until_epoch() -> Result<()> {
        let device = VulkanDevice::new()?;
        if !device.inner.submission_timeline_enabled {
            eprintln!("skipping transient scratch epoch test: timeline semaphore unavailable");
            return Ok(());
        }

        let destination = GpuBuffer::zeros_f32(&device, 4)?;
        let scratch = GpuBuffer::transient_f32(&device, 4)?;
        let scratch_buffer = scratch.allocation.buffer;
        let scratch_offset = scratch.allocation.buffer_offset_bytes;
        let scratch_lease = scratch
            .allocation
            .scratch_lease
            .context("transient scratch did not carry a slab lease")?;
        let mut commands = ComputeBatch::new(&device)?;
        commands.upload_f32(&scratch, &[1.0, 2.0, 3.0, 4.0])?;
        commands.copy_f32(&scratch, &destination, 4)?;
        let submitted = commands.submit_async()?;
        let completion = submitted
            .timeline_wait()
            .context("transient scratch submission did not expose a timeline wait")?;

        // Dropping the last logical view retires only its slab range. The
        // persistent VkBuffer remains device-owned while this offset stays
        // unavailable until the submission epoch completes.
        drop(submitted);
        drop(scratch);
        {
            let arena = device
                .inner
                .submission_resource_arena
                .lock()
                .map_err(|_| anyhow!("Vulkan submission-resource arena lock was poisoned"))?;
            assert!(arena
                .in_flight_scratch_leases
                .iter()
                .any(|retirement| retirement.lease.lease_id == scratch_lease.lease_id));
        }

        completion.wait()?;
        let stats = device.submission_arena_stats()?;
        assert_eq!(stats.in_flight_scratch_leases, 0);
        assert!(stats.timeline_reaped_scratch_leases >= 1);
        let reused = GpuBuffer::transient_f32(&device, 4)?;
        assert_eq!(reused.allocation.buffer, scratch_buffer);
        assert_eq!(reused.allocation.buffer_offset_bytes, scratch_offset);
        assert_eq!(destination.read_f32(4)?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn cloned_device_views_synchronize_shared_queue_and_command_pool_host_access() -> Result<()> {
        let device = VulkanDevice::new()?;
        std::thread::scope(|scope| -> Result<()> {
            let handles = (0..2usize)
                .map(|worker| {
                    let device = device.clone();
                    scope.spawn(move || -> Result<()> {
                        for round in 0..3usize {
                            let values = [
                                worker as f32,
                                round as f32,
                                (worker * 10 + round) as f32,
                                -((worker + round) as f32),
                            ];
                            let source = GpuBuffer::from_f32(&device, &values)?;
                            let destination = GpuBuffer::zeros_f32(&device, values.len())?;
                            let mut commands = ComputeBatch::new(&device)?;
                            commands.copy_f32_range(&source, 0, &destination, 0, values.len())?;
                            commands.submit()?;
                            assert_eq!(destination.read_f32(values.len())?, values);
                        }
                        Ok(())
                    })
                })
                .collect::<Vec<_>>();
            for handle in handles {
                handle.join().map_err(|_| {
                    anyhow::anyhow!("concurrent Vulkan submission worker panicked")
                })??;
            }
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn kernel_profile_categories_keep_optimizer_mirror_costs_distinct() {
        assert_eq!(
            kernel_profile_category("adamw_fp16_mirror"),
            "optimizer.adamw-fp16-mirror"
        );
        assert_eq!(
            kernel_profile_category("fp32_to_fp16_packed"),
            "mixed-precision.mirror-pack"
        );
        assert_eq!(
            kernel_profile_category("fp16_packed_to_fp32"),
            "mixed-precision.mirror-unpack"
        );
        assert_eq!(
            kernel_profile_category("cross_entropy_linear_input_grad_streaming_fp16_packed"),
            "loss.lm-projection"
        );
        assert_eq!(
            kernel_profile_category("rwkv_matrix_state_backward_fused_rkv_add3"),
            "rwkv.recurrent"
        );
    }

    #[test]
    fn device_group_transport_candidate_requires_shared_multi_device_group() {
        let primary = external_memory_test_device(0, "device-a", "driver-a", 4);
        let same_group = external_memory_test_device(1, "device-b", "driver-a", 4);
        let different_group = external_memory_test_device(2, "device-c", "driver-a", 5);

        assert!(primary.device_group_transport_candidate_with(&same_group));
        assert!(!primary.device_group_transport_candidate_with(&different_group));
    }

    #[test]
    fn opaque_external_memory_transport_candidate_requires_matching_uuids() {
        let primary = external_memory_test_device(0, "device-a", "driver-a", 4);
        let same_physical = external_memory_test_device(1, "device-a", "driver-a", 5);
        let different_device = external_memory_test_device(2, "device-b", "driver-a", 4);
        let different_driver = external_memory_test_device(3, "device-a", "driver-b", 4);

        assert!(primary.opaque_external_memory_transport_candidate_with(&same_physical));
        assert!(!primary.opaque_external_memory_transport_candidate_with(&different_device));
        assert!(!primary.opaque_external_memory_transport_candidate_with(&different_driver));
    }

    #[test]
    fn opaque_external_transport_requires_gpu_semaphore_sync() {
        let primary = external_memory_test_device(0, "device-a", "driver-a", 4);
        let mut peer = external_memory_test_device(1, "device-a", "driver-a", 5);
        assert!(primary.opaque_external_transport_candidate_with(&peer));

        peer.external_semaphore.opaque_win32_importable = false;
        peer.external_semaphore.opaque_fd_importable = false;
        assert!(primary.opaque_external_memory_transport_candidate_with(&peer));
        assert!(!primary.opaque_external_transport_candidate_with(&peer));
        assert_eq!(
            primary.gradient_transport_plan_with(&peer).direct_candidate,
            None
        );
    }

    #[test]
    fn gradient_transport_plan_keeps_staging_active_until_direct_probe_exists() {
        let primary = external_memory_test_device(0, "device-a", "driver-a", 4);
        let same_group = external_memory_test_device(1, "device-b", "driver-a", 4);
        let same_physical = external_memory_test_device(2, "device-a", "driver-a", 5);
        let unrelated = external_memory_test_device(3, "device-c", "driver-c", 6);

        let device_group = primary.gradient_transport_plan_with(&same_group);
        assert_eq!(
            device_group.active_backend,
            VulkanGradientTransportBackend::HostVisibleStagedV2Pipelined
        );
        assert_eq!(
            device_group.direct_candidate,
            Some(VulkanGradientTransportBackend::DeviceGroup)
        );

        let external = primary.gradient_transport_plan_with(&same_physical);
        assert_eq!(
            external.direct_candidate,
            Some(VulkanGradientTransportBackend::OpaqueExternalMemory)
        );

        let fallback = primary.gradient_transport_plan_with(&unrelated);
        assert_eq!(fallback.direct_candidate, None);
        assert_eq!(
            fallback.active_backend.label(),
            "host-visible-staged-v2-pipelined"
        );
    }

    #[test]
    fn memory_pressure_bucket_is_coarse_bounded_context() {
        let mut budget = VulkanMemoryBudget {
            device_local_budget_bytes: 800,
            ..Default::default()
        };
        assert_eq!(budget.device_local_pressure_bucket(), Some(0));
        budget.device_local_usage_bytes = 200;
        assert_eq!(budget.device_local_pressure_bucket(), Some(2));
        budget.device_local_usage_bytes = 799;
        assert_eq!(budget.device_local_pressure_bucket(), Some(7));
        budget.device_local_usage_bytes = 1_600;
        assert_eq!(budget.device_local_pressure_bucket(), Some(7));
        budget.device_local_budget_bytes = 0;
        assert_eq!(budget.device_local_pressure_bucket(), None);
    }

    #[test]
    fn pooled_range_allocation_respects_alignment_and_preserves_slack() {
        let mut free_ranges = vec![MemoryRange {
            offset: 3,
            size: 64,
        }];

        let offset = take_aligned_range(&mut free_ranges, 8, 16);

        assert_eq!(offset, Some(16));
        assert_eq!(
            free_ranges,
            vec![
                MemoryRange {
                    offset: 3,
                    size: 13,
                },
                MemoryRange {
                    offset: 24,
                    size: 43,
                },
            ]
        );
    }

    #[test]
    fn pooled_range_free_coalesces_adjacent_and_overlapping_ranges() {
        let mut free_ranges = vec![
            MemoryRange {
                offset: 0,
                size: 16,
            },
            MemoryRange {
                offset: 32,
                size: 16,
            },
        ];

        insert_free_range(
            &mut free_ranges,
            MemoryRange {
                offset: 16,
                size: 20,
            },
        );

        assert_eq!(
            free_ranges,
            vec![MemoryRange {
                offset: 0,
                size: 48,
            }]
        );
    }
}
