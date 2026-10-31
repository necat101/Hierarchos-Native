//! GPU regression/benchmark for exact-order reduction optimizations.
//!
//! Pass a directory containing the previous versions of the named SPIR-V
//! files. Example: cargo run --release --bin reduction_kernel_validation --
//! target/shader-reduction-baseline
//!
//! `--candidate-dir` overrides the directory holding the new modules (default
//! `shaders`), which lets the same harness compare two historical artifacts.
//! Every compared buffer is fully scanned: mismatches are collected across all
//! cases and reported together instead of stopping at the first one.
//!
//! `--write-golden PATH` records a digest of the candidate outputs per
//! invocation, and `--golden PATH` re-verifies them. In either golden mode the
//! baseline directory is optional, because the digests are the baseline: CI can
//! then catch an optimization regression without shipping old artifacts.
#![allow(dead_code)]

#[path = "../kernel_golden.rs"]
mod kernel_golden;
#[path = "../vulkan.rs"]
mod vulkan;

use anyhow::{bail, Context, Result};
use kernel_golden::{coverage_args, Golden};
use std::{
    fs,
    path::{Path, PathBuf},
};
use vulkan::{BindingAccess, ComputeBatch, ComputeKernel, GpuBuffer, VulkanDevice};

fn signal(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            (((i as u32).wrapping_mul(1664525).wrapping_add(seed) >> 10) % 4093) as f32 / 2048.0
                - 1.0
        })
        .collect()
}

/// Baseline kernels are optional: golden mode verifies against recorded
/// digests, so no previous artifact directory is needed.
fn kernel_pair(
    device: &VulkanDevice,
    baseline: Option<&Path>,
    candidate: &Path,
    name: &str,
    accesses: &[BindingAccess],
    push_bytes: u32,
) -> Result<(Option<ComputeKernel>, ComputeKernel)> {
    let filename = format!("{name}.spv");
    let old = baseline
        .map(|dir| {
            let bytes = fs::read(dir.join(&filename))
                .with_context(|| format!("reading baseline {filename}"))?;
            ComputeKernel::new_with_access(device, &bytes, accesses, push_bytes)
        })
        .transpose()?;
    let new = fs::read(candidate.join(&filename))
        .with_context(|| format!("reading candidate {filename}"))?;
    Ok((
        old,
        ComputeKernel::new_with_access(device, &new, accesses, push_bytes)?,
    ))
}

fn dispatch(
    device: &VulkanDevice,
    kernel: &ComputeKernel,
    buffers: &[GpuBuffer],
    push: &[u32],
    groups: u32,
) -> Result<()> {
    let mut batch = ComputeBatch::new(device)?;
    kernel.record_dispatch(
        &mut batch,
        &buffers.iter().collect::<Vec<_>>(),
        bytemuck::cast_slice(push),
        [groups, 1, 1],
    )?;
    batch.submit()
}

#[derive(Default)]
struct Mismatches(Vec<String>);

impl Mismatches {
    fn record(
        &mut self,
        label: &str,
        left: &GpuBuffer,
        right: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        let a = left.read_f32(len)?;
        let b = right.read_f32(len)?;
        let mut differing = 0usize;
        let mut first = None;
        for (index, (&a, &b)) in a.iter().zip(&b).enumerate() {
            if a.to_bits() != b.to_bits() {
                differing += 1;
                if first.is_none() {
                    first = Some((index, a, b));
                }
            }
        }
        if let Some((index, a, b)) = first {
            self.0.push(format!(
                "{label}: {differing}/{len} elements differ; first at {index}: {a:?} ({:08x}) != {b:?} ({:08x})",
                a.to_bits(),
                b.to_bits()
            ));
        }
        Ok(())
    }
}

fn time(
    device: &VulkanDevice,
    kernel: &ComputeKernel,
    buffers: &[GpuBuffer],
    push: &[u32],
    groups: u32,
) -> Result<f64> {
    let refs = buffers.iter().collect::<Vec<_>>();
    let mut samples = Vec::new();
    for _ in 0..5 {
        let ms = device.time_compute_batch_ms(|batch| {
            for _ in 0..8 {
                kernel.record_dispatch(batch, &refs, bytemuck::cast_slice(push), [groups, 1, 1])?;
            }
            Ok(())
        })?;
        samples.push(ms / 8.0);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[2])
}

fn qwen(
    device: &VulkanDevice,
    baseline: Option<&Path>,
    candidate: &Path,
    mismatches: &mut Mismatches,
    golden: &mut Option<Golden>,
) -> Result<usize> {
    use BindingAccess::{ReadOnly as R, WriteOnly as W};
    let (old_kernel, new_kernel) = kernel_pair(
        device,
        baseline,
        candidate,
        "transformer_qwen_gdn_gate_norm_param_grad",
        &[R, R, R, W],
        16,
    )?;
    let cases = [
        (1, 1, 1),
        (7, 3, 7),
        (21, 3, 32),
        (32, 2, 63),
        (65, 1, 64),
        (43, 3, 65),
        (128, 4, 128),
        (256, 4, 192),
    ];
    for &(rows, heads, dim) in &cases {
        let len = (rows * heads * dim) as usize;
        let label = format!("qwen gated RMS weight gradient rows={rows} heads={heads} dim={dim}");
        let inputs = [signal(len, 101), signal(len, 701), signal(len, 1701)];
        let mut new = inputs
            .iter()
            .map(|x| GpuBuffer::from_f32(device, x))
            .collect::<Result<Vec<_>>>()?;
        new.push(GpuBuffer::zeros_f32(device, dim as usize)?);
        let old = if old_kernel.is_some() {
            let mut buffers = inputs
                .iter()
                .map(|x| GpuBuffer::from_f32(device, x))
                .collect::<Result<Vec<_>>>()?;
            buffers.push(GpuBuffer::zeros_f32(device, dim as usize)?);
            Some(buffers)
        } else {
            None
        };
        let push = [rows, heads, dim, 1.0e-6f32.to_bits()];
        let groups = dim.div_ceil(64);
        dispatch(device, &new_kernel, &new, &push, groups)?;
        if let (Some(kernel), Some(buffers)) = (&old_kernel, old.as_ref()) {
            dispatch(device, kernel, buffers, &push, groups)?;
            mismatches.record(&label, &buffers[3], &new[3], dim as usize)?;
        }
        if let Some(golden) = golden.as_mut() {
            golden.observe(&label, &new[3].read_f32(dim as usize)?);
        }
        if let Some(kernel) = &old_kernel {
            if dim >= 128 {
                let old_ms = time(device, kernel, old.as_ref().unwrap(), &push, groups)?;
                let new_ms = time(device, &new_kernel, &new, &push, groups)?;
                println!("qwen rows={rows} heads={heads} dim={dim} old_ms={old_ms:.6} new_ms={new_ms:.6} speedup={:.3}", old_ms / new_ms);
            }
        }
    }
    Ok(cases.len())
}

fn rms(
    device: &VulkanDevice,
    baseline: Option<&Path>,
    candidate: &Path,
    mismatches: &mut Mismatches,
    golden: &mut Option<Golden>,
) -> Result<usize> {
    use BindingAccess::{ReadOnly as R, WriteOnly as W};
    let mut checked = 0;
    for residual in [false, true] {
        let name = if residual {
            "transformer_rms_norm_input_grad_residual"
        } else {
            "transformer_rms_norm_input_grad"
        };
        let accesses = if residual {
            vec![R, R, R, R, R, R, W, W]
        } else {
            vec![R, R, R, R, R, W]
        };
        let (old_kernel, new_kernel) =
            kernel_pair(device, baseline, candidate, name, &accesses, 28)?;
        for &(rows, dim) in &[(1u32, 1u32), (3, 7), (65, 32), (7, 63), (128, 256)] {
            let len = (rows * dim) as usize;
            let input = signal(len, 101);
            let grad = signal(len, 351);
            let weight = signal((dim * 3) as usize, 701);
            let mean = vec![0.351f32; rows as usize];
            let rstd = vec![1.0f32 / 0.351f32.sqrt(); rows as usize];
            let mut inputs = vec![grad, input, weight, rstd, mean];
            if residual {
                inputs.push(signal(len, 1901));
            }
            let outputs = if residual { 2 } else { 1 };
            for pow in 0..=1u32 {
                for products in 0..=3u32 {
                    for exact in 0..=1u32 {
                        let label = format!("{name} rows={rows} dim={dim} pow={pow} products={products} exact={exact}");
                        let build = |device: &VulkanDevice| -> Result<Vec<GpuBuffer>> {
                            let mut buffers = inputs
                                .iter()
                                .map(|x| GpuBuffer::from_f32(device, x))
                                .collect::<Result<Vec<_>>>()?;
                            for _ in 0..outputs {
                                buffers.push(GpuBuffer::zeros_f32(device, len)?);
                            }
                            Ok(buffers)
                        };
                        let new = build(device)?;
                        let push = [rows, dim, 3, 1.0f32.to_bits(), pow, products, exact];
                        let groups = rows.div_ceil(64);
                        dispatch(device, &new_kernel, &new, &push, groups)?;
                        if let Some(kernel) = &old_kernel {
                            let old = build(device)?;
                            dispatch(device, kernel, &old, &push, groups)?;
                            for index in inputs.len()..old.len() {
                                mismatches.record(
                                    &format!("{label} output={index}"),
                                    &old[index],
                                    &new[index],
                                    len,
                                )?;
                            }
                        }
                        if let Some(golden) = golden.as_mut() {
                            let mut parts = Vec::with_capacity(outputs);
                            let mut owned = Vec::with_capacity(outputs);
                            for index in inputs.len()..new.len() {
                                owned.push(new[index].read_f32(len)?);
                            }
                            for values in &owned {
                                parts.push(values.as_slice());
                            }
                            golden.observe_parts(&label, &parts);
                        }
                        checked += 1;
                        if rows == 128 && pow == 1 && products == 2 && exact == 0 {
                            if let Some(kernel) = &old_kernel {
                                let old = build(device)?;
                                let old_ms = time(device, kernel, &old, &push, groups)?;
                                let new_ms = time(device, &new_kernel, &new, &push, groups)?;
                                println!("{name} rows={rows} dim={dim} old_ms={old_ms:.6} new_ms={new_ms:.6} speedup={:.3}", old_ms / new_ms);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(checked)
}

fn adamw(
    device: &VulkanDevice,
    baseline: Option<&Path>,
    candidate: &Path,
    mismatches: &mut Mismatches,
    golden: &mut Option<Golden>,
) -> Result<usize> {
    use BindingAccess::{ReadOnly as R, ReadWrite as RW};
    let (old_kernel, new_kernel) =
        kernel_pair(device, baseline, candidate, "adamw", &[RW, R, RW, RW], 28)?;
    let mut checked = 0;
    for len in [1u32, 255, 256, 257, 65537] {
        for step in [1u32, 17, 4096] {
            let label = format!("adamw len={len} step={step}");
            let inputs = [
                signal(len as usize, 101),
                signal(len as usize, 701),
                signal(len as usize, 1701),
                vec![0.15; len as usize],
            ];
            let build = |device: &VulkanDevice| -> Result<Vec<GpuBuffer>> {
                inputs
                    .iter()
                    .map(|x| GpuBuffer::from_f32(device, x))
                    .collect::<Result<Vec<_>>>()
            };
            let new = build(device)?;
            let push = [
                len,
                step,
                0.001f32.to_bits(),
                0.9f32.to_bits(),
                0.999f32.to_bits(),
                1.0e-8f32.to_bits(),
                0.01f32.to_bits(),
            ];
            let groups = len.div_ceil(256);
            dispatch(device, &new_kernel, &new, &push, groups)?;
            if let Some(kernel) = &old_kernel {
                let old = build(device)?;
                dispatch(device, kernel, &old, &push, groups)?;
                for index in [0, 2, 3] {
                    mismatches.record(
                        &format!("{label} output={index}"),
                        &old[index],
                        &new[index],
                        len as usize,
                    )?;
                }
            }
            if let Some(golden) = golden.as_mut() {
                let mut parts = Vec::new();
                let mut owned = Vec::new();
                for index in [0, 2, 3] {
                    owned.push(new[index].read_f32(len as usize)?);
                }
                for values in &owned {
                    parts.push(values.as_slice());
                }
                golden.observe_parts(&label, &parts);
            }
            checked += 1;
            if len == 65537 && step == 17 {
                if let Some(kernel) = &old_kernel {
                    let old = build(device)?;
                    let old_ms = time(device, kernel, &old, &push, groups)?;
                    let new_ms = time(device, &new_kernel, &new, &push, groups)?;
                    println!(
                        "adamw len={len} old_ms={old_ms:.6} new_ms={new_ms:.6} speedup={:.3}",
                        old_ms / new_ms
                    );
                }
            }
        }
    }
    Ok(checked)
}

fn main() -> Result<()> {
    let mut baseline = None;
    let mut candidate = PathBuf::from("shaders");
    let mut golden_compare = None;
    let mut golden_write = None;
    let mut golden_strict = false;
    let arguments = std::env::args().skip(1).collect::<Vec<String>>();
    let mut args = arguments.iter().cloned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--candidate-dir" => {
                candidate = PathBuf::from(args.next().context("missing candidate directory")?);
            }
            "--golden" => {
                golden_compare = Some(PathBuf::from(args.next().context("missing golden file")?));
            }
            "--write-golden" => {
                golden_write = Some(PathBuf::from(args.next().context("missing golden file")?));
            }
            "--golden-strict" => golden_strict = true,
            // Resolved by `coverage_args` below; consumed here so the harness does not read
            // the flag's value as its own argument.
            "--coverage" | "--coverage-key" => {
                let _ = args.next().context("missing coverage argument")?;
            }
            "--record-coverage" | "--allow-coverage-shrink" | "--bootstrap-coverage" => {}
            flag if flag.starts_with("--") => bail!("unknown argument {flag}"),
            _ => baseline = Some(PathBuf::from(arg)),
        }
    }
    if golden_compare.is_some() && golden_write.is_some() {
        bail!("--golden and --write-golden are mutually exclusive");
    }
    if baseline.is_none() && golden_compare.is_none() && golden_write.is_none() {
        bail!(
            "usage: reduction_kernel_validation BASELINE_SPV_DIRECTORY [--candidate-dir DIR]\n\
             or:    reduction_kernel_validation --write-golden FILE | --golden FILE"
        );
    }
    let device = VulkanDevice::new()?;
    println!("device={}", device.name());
    if golden_strict && golden_compare.is_none() {
        bail!("--golden-strict needs a --golden file to verify against");
    }
    let coverage = coverage_args(&arguments)?;
    let mut golden = match (&golden_compare, &golden_write) {
        (Some(path), _) => Some(
            Golden::compare(path, device.name())?
                .strict(golden_strict)
                .with_coverage(coverage, path),
        ),
        (None, Some(path)) => {
            Some(Golden::write(path, device.name())?.with_coverage(coverage, path))
        }
        (None, None) => None,
    };
    let mut mismatches = Mismatches::default();
    let baseline = baseline.as_deref();
    let checked = qwen(&device, baseline, &candidate, &mut mismatches, &mut golden)?
        + rms(&device, baseline, &candidate, &mut mismatches, &mut golden)?
        + adamw(&device, baseline, &candidate, &mut mismatches, &mut golden)?;
    if !mismatches.0.is_empty() {
        for mismatch in &mismatches.0 {
            eprintln!("mismatch {mismatch}");
        }
        bail!(
            "{} compared outputs differ from the baseline",
            mismatches.0.len()
        );
    }
    println!("PASS: {checked} GPU cases, all compared outputs bit-identical");
    if let Some(golden) = golden {
        golden.finish()?;
    }
    Ok(())
}
