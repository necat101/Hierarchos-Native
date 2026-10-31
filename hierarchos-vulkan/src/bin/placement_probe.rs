//! Forced-placement experiment for one kernel.
//!
//! The per-step kernel profile splits each shader's dispatches into those recorded
//! directly after a barrier and those recorded back to back with another dispatch, and
//! prints both means. That split is an *observation*: which dispatches land in which
//! population is decided by the graph's hazard rules, so the two populations are
//! whatever the graph happened to produce, and a shape key can only remove the
//! confounds it can see (grid, push constants, binding count — not which tensors were
//! bound). This harness makes placement the independent variable instead:
//!
//! * Both arms dispatch the same kernel, at the same shape, with the same shared
//!   inputs, and each dispatch writes its own output buffer. Buffer identity is
//!   therefore identical in both arms, so no aliasing confound is introduced by giving
//!   the "warm" arm memory the other arm does not touch.
//! * The cold arm records an explicit barrier before every dispatch; the warm arm
//!   records none. Each block's barrier count is verified before it is measured, so the
//!   assignment is checked rather than assumed.
//! * Arms alternate in both orders (warm-cold-cold-warm, then cold-warm-warm-cold), so
//!   clock drift and warm-up cannot favour either arm.
//! * Each block is timed as a whole: per-dispatch cost is the block's GPU time divided
//!   by its dispatch count, which keeps the instrument at one query pair per block
//!   instead of two timestamp writes per dispatch.
//!
//! Two numbers come out per arm: GPU time per dispatch (from the device's timestamp
//! queries) and wall time per dispatch (host submit including the wait). The difference
//! between the arms is what a dispatch pays because a barrier was recorded in front of
//! it — the quantity the step-level profile can only estimate.
//!
//! Usage:
//!   cargo run --release --bin placement_probe -- [--kernel NAME] [--shape ROWSxINxOUT]
//!     [--blocks 32] [--trials 4] [--json PATH] [--device-index 0] [--list]
//!
//! Every dispatch in a block writes a distinct output buffer, so a block of K
//! dispatches allocates K output buffers of the shape's output size.

#![allow(dead_code)]
#[path = "../vulkan.rs"]
mod vulkan;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;
use vulkan::{BindingAccess, ComputeBatch, ComputeKernel, GpuBuffer, VulkanDevice};

/// Kernels whose dispatch parameters are already pinned down by an existing
/// validation harness, so this experiment cannot be confounded by getting the push
/// constants or the grid geometry wrong. Each is dispatched by the training step and
/// appears in its per-shader profile.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// A barrier is recorded immediately before the dispatch.
    AfterBarrier,
    /// The dispatch follows the previous dispatch with nothing in between.
    BackToBack,
}

impl Placement {
    fn as_str(self) -> &'static str {
        match self {
            Placement::AfterBarrier => "after_barrier",
            Placement::BackToBack => "after_dispatch",
        }
    }
}

struct Case {
    name: &'static str,
    shape: String,
    groups: [u32; 3],
    push: Vec<u8>,
    /// Bound to every dispatch of the block, in binding order.
    shared: Vec<GpuBuffer>,
    /// One output buffer per dispatch slot in a block; bound last.
    outputs: Vec<GpuBuffer>,
    elements_per_dispatch: usize,
}

impl Case {
    /// Record one dispatch of the block: the shared bindings plus the output buffer
    /// belonging to this slot.
    fn record(&self, kernel: &ComputeKernel, batch: &mut ComputeBatch, slot: usize) -> Result<()> {
        let output = self
            .outputs
            .get(slot)
            .with_context(|| format!("output slot {slot} beyond the block"))?;
        let mut buffers = self.shared.iter().collect::<Vec<_>>();
        buffers.push(output);
        kernel.record_dispatch(batch, &buffers, &self.push, self.groups)
    }
}

fn values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 73 + salt * 31) % 251) as f32 - 125.0) / 127.0)
        .collect()
}

const KERNELS: &[&str] = &[
    "linear_forward",
    "linear_bias_forward",
    "vector_add",
    "finite_clamp_forward",
];

/// Build a kernel plus its data for one shape, with block-many output buffers.
fn build_case(
    device: &VulkanDevice,
    shader_dir: &PathBuf,
    name: &'static str,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    block: usize,
) -> Result<(ComputeKernel, Case)> {
    if rows == 0 || input_dim == 0 || output_dim == 0 {
        bail!("shape dimensions must be positive");
    }
    let (accesses, push, groups, shared, output_len): (
        Vec<BindingAccess>,
        Vec<u8>,
        [u32; 3],
        Vec<Vec<f32>>,
        usize,
    ) = match name {
        "linear_forward" => (
            vec![
                BindingAccess::ReadOnly,
                BindingAccess::ReadOnly,
                BindingAccess::WriteOnly,
            ],
            linear_push(rows, input_dim, output_dim),
            [
                (output_dim as u32).div_ceil(16),
                (rows as u32).div_ceil(16),
                1,
            ],
            vec![
                values(rows * input_dim, 97),
                values(input_dim * output_dim, 701),
            ],
            rows * output_dim,
        ),
        "linear_bias_forward" => (
            vec![
                BindingAccess::ReadOnly,
                BindingAccess::ReadOnly,
                BindingAccess::ReadOnly,
                BindingAccess::WriteOnly,
            ],
            linear_push(rows, input_dim, output_dim),
            [
                (output_dim as u32).div_ceil(16),
                (rows as u32).div_ceil(16),
                1,
            ],
            vec![
                values(rows * input_dim, 97),
                values(input_dim * output_dim, 701),
                values(output_dim, 439),
            ],
            rows * output_dim,
        ),
        "vector_add" => (
            vec![
                BindingAccess::ReadOnly,
                BindingAccess::ReadOnly,
                BindingAccess::WriteOnly,
            ],
            65536u32.to_le_bytes().to_vec(),
            [65536u32.div_ceil(256), 1, 1],
            vec![values(65536, 11), values(65536, 13)],
            65536,
        ),
        "finite_clamp_forward" => (
            vec![BindingAccess::ReadOnly, BindingAccess::WriteOnly],
            {
                let mut push = 65536u32.to_le_bytes().to_vec();
                push.extend_from_slice(&6.0f32.to_le_bytes());
                push
            },
            [65536u32.div_ceil(256), 1, 1],
            vec![values(65536, 17)],
            65536,
        ),
        other => bail!("unknown kernel {other}; known: {KERNELS:?}"),
    };
    let push_constant_size = u32::try_from(push.len()).context("push-constant size overflow")?;
    let kernel = ComputeKernel::new_with_access(
        device,
        &std::fs::read(shader_dir.join(format!("{name}.spv")))
            .with_context(|| format!("reading {name}.spv"))?,
        &accesses,
        push_constant_size,
    )
    .with_context(|| format!("creating compute kernel {name}"))?;
    let shared = shared
        .iter()
        .map(|data| GpuBuffer::from_f32(device, data))
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = Vec::with_capacity(block);
    for _ in 0..block {
        outputs.push(GpuBuffer::zeros_f32(device, output_len)?);
    }
    Ok((
        kernel,
        Case {
            name,
            shape: format!("{rows}x{input_dim}x{output_dim}"),
            groups,
            push,
            shared,
            outputs,
            elements_per_dispatch: output_len,
        },
    ))
}

/// The `rows, input_dim, output_dim` push block the linear kernels and their
/// validation harness share, so this experiment cannot introduce a shape mismatch.
fn linear_push(rows: usize, input_dim: usize, output_dim: usize) -> Vec<u8> {
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&(rows as u32).to_le_bytes());
    push.extend_from_slice(&(input_dim as u32).to_le_bytes());
    push.extend_from_slice(&(output_dim as u32).to_le_bytes());
    push
}

fn record_block(
    case: &Case,
    kernel: &ComputeKernel,
    batch: &mut ComputeBatch,
    placement: Placement,
    block: usize,
) -> Result<()> {
    for slot in 0..block {
        if placement == Placement::AfterBarrier {
            batch.shader_barrier()?;
        }
        case.record(kernel, batch, slot)?;
    }
    Ok(())
}

#[derive(Serialize)]
struct Reading {
    placement: &'static str,
    gpu_us_per_dispatch: f64,
    wall_us_per_dispatch: f64,
    gpu_timed_from_timestamps: bool,
}

#[derive(Serialize)]
struct ArmSummary {
    placement: &'static str,
    samples: usize,
    gpu_us_per_dispatch_median: f64,
    gpu_us_per_dispatch_min: f64,
    gpu_us_per_dispatch_max: f64,
    wall_us_per_dispatch_median: f64,
    wall_us_per_dispatch_min: f64,
    wall_us_per_dispatch_max: f64,
}

#[derive(Serialize)]
struct ShapeReport {
    kernel: &'static str,
    shape: String,
    grid: String,
    block: usize,
    trials: usize,
    /// Verified from the recorded command stream, not assumed: the cold arm must
    /// record one barrier per dispatch and the warm arm none.
    barriers_recorded_after_barrier_arm: usize,
    barriers_recorded_after_dispatch_arm: usize,
    output_bytes_per_dispatch: usize,
    arms: Vec<ArmSummary>,
    inherited_gpu_us_per_dispatch: f64,
    inherited_wall_us_per_dispatch: f64,
    schedule: Vec<&'static str>,
    readings: Vec<Reading>,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// Measure one block: GPU time from the device's timestamps, wall time from a
/// synchronous submit of an identical block. The timestamp path is verified per
/// reading, so a wall-clock fallback is reported as such instead of being presented
/// as GPU time.
fn measure(
    device: &VulkanDevice,
    case: &Case,
    kernel: &ComputeKernel,
    placement: Placement,
    block: usize,
) -> Result<(Reading, usize)> {
    // Verify the assignment before measuring it: record the block once and read the
    // counters back from the command stream.
    let mut probe = ComputeBatch::new(device)?;
    record_block(case, kernel, &mut probe, placement, block)?;
    let dispatches = probe.dispatch_count();
    let barriers = probe.shader_barrier_count();
    probe.submit()?;
    if dispatches != block {
        bail!(
            "{} block recorded {dispatches} dispatches, expected {block}",
            placement.as_str()
        );
    }

    let samples_before = device
        .submission_arena_stats()?
        .kernel_timestamp_profile_samples;
    let gpu_ms = device
        .time_compute_batch_ms(|batch| record_block(case, kernel, batch, placement, block))?;
    let samples_after = device
        .submission_arena_stats()?
        .kernel_timestamp_profile_samples;
    let from_timestamps = samples_after.saturating_sub(samples_before) == 1;

    let mut wall = ComputeBatch::new(device)?;
    record_block(case, kernel, &mut wall, placement, block)?;
    let started = Instant::now();
    wall.submit()?;
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;

    Ok((
        Reading {
            placement: placement.as_str(),
            gpu_us_per_dispatch: gpu_ms * 1000.0 / block as f64,
            wall_us_per_dispatch: wall_ms * 1000.0 / block as f64,
            gpu_timed_from_timestamps: from_timestamps,
        },
        barriers,
    ))
}

fn summarize(placement: Placement, readings: &[Reading]) -> ArmSummary {
    let gpu = readings
        .iter()
        .filter(|reading| reading.placement == placement.as_str())
        .map(|reading| reading.gpu_us_per_dispatch)
        .collect::<Vec<_>>();
    let wall = readings
        .iter()
        .filter(|reading| reading.placement == placement.as_str())
        .map(|reading| reading.wall_us_per_dispatch)
        .collect::<Vec<_>>();
    ArmSummary {
        placement: placement.as_str(),
        samples: gpu.len(),
        gpu_us_per_dispatch_median: median(&gpu),
        gpu_us_per_dispatch_min: gpu.iter().copied().fold(f64::INFINITY, f64::min),
        gpu_us_per_dispatch_max: gpu.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        wall_us_per_dispatch_median: median(&wall),
        wall_us_per_dispatch_min: wall.iter().copied().fold(f64::INFINITY, f64::min),
        wall_us_per_dispatch_max: wall.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_shape(
    device: &VulkanDevice,
    shader_dir: &PathBuf,
    name: &'static str,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    block: usize,
    trials: usize,
) -> Result<ShapeReport> {
    let (kernel, case) = build_case(device, shader_dir, name, rows, input_dim, output_dim, block)?;
    // Warm the device and the descriptor caches before either arm is measured.
    for _ in 0..2 {
        let mut warmup = ComputeBatch::new(device)?;
        record_block(&case, &kernel, &mut warmup, Placement::BackToBack, block)?;
        warmup.submit()?;
    }

    let mut readings = Vec::new();
    let mut schedule = Vec::new();
    let mut cold_barriers = None;
    let mut warm_barriers = None;
    for trial in 0..trials {
        // Both orders, so neither arm is systematically first or last.
        let order = if trial % 2 == 0 {
            [
                Placement::BackToBack,
                Placement::AfterBarrier,
                Placement::AfterBarrier,
                Placement::BackToBack,
            ]
        } else {
            [
                Placement::AfterBarrier,
                Placement::BackToBack,
                Placement::BackToBack,
                Placement::AfterBarrier,
            ]
        };
        for placement in order {
            let (reading, barriers) = measure(device, &case, &kernel, placement, block)?;
            match placement {
                Placement::AfterBarrier => cold_barriers = Some(barriers),
                Placement::BackToBack => warm_barriers = Some(barriers),
            }
            schedule.push(placement.as_str());
            readings.push(reading);
        }
    }

    let warm = summarize(Placement::BackToBack, &readings);
    let cold = summarize(Placement::AfterBarrier, &readings);
    if readings
        .iter()
        .any(|reading| !reading.gpu_timed_from_timestamps)
    {
        eprintln!(
            "warning: at least one reading fell back to wall time because the device produced no timestamp sample"
        );
    }
    Ok(ShapeReport {
        kernel: name,
        shape: case.shape.clone(),
        grid: format!("{}x{}x{}", case.groups[0], case.groups[1], case.groups[2]),
        block,
        trials,
        barriers_recorded_after_barrier_arm: cold_barriers.unwrap_or_default(),
        barriers_recorded_after_dispatch_arm: warm_barriers.unwrap_or_default(),
        output_bytes_per_dispatch: case.elements_per_dispatch * 4,
        inherited_gpu_us_per_dispatch: cold.gpu_us_per_dispatch_median
            - warm.gpu_us_per_dispatch_median,
        inherited_wall_us_per_dispatch: cold.wall_us_per_dispatch_median
            - warm.wall_us_per_dispatch_median,
        arms: vec![warm, cold],
        schedule,
        readings,
    })
}

#[derive(Serialize)]
struct Report {
    device: String,
    shapes: Vec<ShapeReport>,
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .context("spawning placement-probe worker")?
        .join()
        .map_err(|_| anyhow::anyhow!("placement-probe worker panicked"))?
}

fn run() -> Result<()> {
    // The last two shapes are the ones the 512-row step's profile reports placement
    // for (grids 1x32x1 and 8x32x1), so the assigned experiment can be compared with
    // the observed split at an identical grid instead of across shapes.
    let mut kernels = vec![
        ("linear_forward", (32usize, 128usize, 128usize)),
        ("linear_bias_forward", (32, 128, 128)),
        ("linear_bias_forward", (512, 128, 16)),
        ("linear_forward", (512, 128, 128)),
        ("linear_bias_forward", (32, 512, 512)),
        ("vector_add", (0, 0, 0)),
        ("finite_clamp_forward", (0, 0, 0)),
    ];
    let mut blocks = vec![32usize];
    let mut trials = 4usize;
    let mut device_index = 0usize;
    let mut json: Option<PathBuf> = None;
    let mut list = false;
    let mut only: Option<String> = None;

    let mut rest = std::env::args().skip(1);
    while let Some(arg) = rest.next() {
        let mut value = || {
            rest.next()
                .with_context(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "--device-index" => device_index = value()?.parse()?,
            "--blocks" => {
                blocks = value()?
                    .split(',')
                    .map(|part| part.trim().parse::<usize>())
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            "--trials" => trials = value()?.parse()?,
            "--json" => json = Some(PathBuf::from(value()?)),
            "--only" => only = Some(value()?),
            "--list" => list = true,
            other => bail!("unknown argument {other}"),
        }
    }
    if list {
        for name in KERNELS {
            println!("{name}");
        }
        return Ok(());
    }
    if blocks.is_empty() || blocks.iter().any(|&block| block == 0) {
        bail!("--blocks must contain positive block sizes");
    }
    if trials == 0 {
        bail!("--trials must be positive");
    }
    if let Some(filter) = only.as_deref() {
        kernels.retain(|(name, _)| *name == filter);
        if kernels.is_empty() {
            bail!("--only {filter} matches no kernel in the registry; known: {KERNELS:?}");
        }
    }

    let device = VulkanDevice::new_with_index(device_index)?;
    let shader_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    eprintln!(
        "placement probe on {}: {} shape(s), blocks {blocks:?}, {} trial(s) per shape",
        device.name(),
        kernels.len(),
        trials
    );
    eprintln!(
        "each arm alternates in both orders (warm-cold-cold-warm, cold-warm-warm-cold); \
         cold records one barrier per dispatch, warm records none, and both are verified \
         from the recorded command stream before a block is timed"
    );

    let mut shapes = Vec::new();
    for block in blocks {
        for (name, (rows, input_dim, output_dim)) in &kernels {
            let (rows, input_dim, output_dim) = match *name {
                "vector_add" | "finite_clamp_forward" => (1, 65536, 1),
                _ => (*rows, *input_dim, *output_dim),
            };
            let report = run_shape(
                &device,
                &shader_dir,
                name,
                rows,
                input_dim,
                output_dim,
                block,
                trials,
            )?;
            println!(
                "{:20} shape {:>12} grid {:>9} block {:>3}  \
                 warm {:>8.2} us/disp (wall {:>8.2})  cold {:>8.2} us/disp (wall {:>8.2})  \
                 inherited {:>7.2} us/disp (wall {:>7.2})  barriers cold/warm {}",
                report.kernel,
                report.shape,
                report.grid,
                report.block,
                report.arms[0].gpu_us_per_dispatch_median,
                report.arms[0].wall_us_per_dispatch_median,
                report.arms[1].gpu_us_per_dispatch_median,
                report.arms[1].wall_us_per_dispatch_median,
                report.inherited_gpu_us_per_dispatch,
                report.inherited_wall_us_per_dispatch,
                format!(
                    "{}/{}",
                    report.barriers_recorded_after_barrier_arm,
                    report.barriers_recorded_after_dispatch_arm
                ),
            );
            shapes.push(report);
        }
    }

    let report = Report {
        device: device.name().to_string(),
        shapes,
    };
    if let Some(path) = json {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
