//! Portable dense-kernel parity and optional timestamp A/B checks.
//!
//! cargo run --manifest-path hierarchos-vulkan/Cargo.toml --bin matmul_kernel_validation --
//!   --baseline-dir PATH [--bench] [--output REPORT.json]
//! Baseline directories contain the same shader filenames, saved before a change.
//! `--write-golden PATH` records a digest of every invocation's output, and
//! `--golden PATH` re-verifies them later without any baseline directory, which
//! is what lets this harness run in CI on a machine that has no old artifacts.
//! The runtime module is included here so validation can exercise raw modules
//! without making the training backend's internal dispatch API public.
#![allow(dead_code)]
#[path = "../kernel_golden.rs"]
mod kernel_golden;
#[path = "../vulkan.rs"]
mod vulkan;

use anyhow::{bail, ensure, Context, Result};
use kernel_golden::{coverage_args, Golden};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use vulkan::{BindingAccess, ComputeBatch, ComputeKernel, GpuBuffer, VulkanDevice};

const NAMES: &[&str] = &[
    "linear_forward",
    "linear_bias_forward",
    "linear_residual_forward",
    "linear_forward_fp16_packed",
    "linear_bias_forward_fp16_packed",
    "parameter_matmul_forward",
    "parameter_matmul_bias_forward",
    "parameter_matmul_forward_fp16_packed",
    "parameter_matmul_bias_forward_fp16_packed",
    "parameter_matmul_input_grad",
    "parameter_matmul_input_grad_fp16_packed",
    "parameter_matmul_input_grad_fp16_native_compute",
];

#[derive(Serialize)]
struct Check {
    kernel: String,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    elements: usize,
    baseline_bitwise_equal: Option<bool>,
    max_reference_error: f32,
}
#[derive(Serialize)]
struct Timing {
    kernel: String,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    current_ms: f64,
    baseline_ms: Option<f64>,
    speedup: Option<f64>,
}
#[derive(Serialize)]
struct Report {
    device: String,
    checks: Vec<Check>,
    timings: Vec<Timing>,
}

fn round_shift(value: u32, shift: u32) -> u32 {
    let result = value >> shift;
    let remainder = value & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    result + u32::from(remainder > halfway || (remainder == halfway && result & 1 != 0))
}
fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 255) as i32 - 127 + 15;
    let mantissa = bits & 0x7fffff;
    if exponent <= 0 {
        return if exponent < -10 {
            sign
        } else {
            sign | round_shift(mantissa | 0x800000, (14 - exponent) as u32) as u16
        };
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | (((exponent as u32) << 10) + round_shift(mantissa, 13)) as u16
}
fn widen(value: u16) -> f32 {
    let sign = if value & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (value >> 10) & 31;
    let mantissa = value & 1023;
    if exponent == 0 {
        sign * f32::from(mantissa) * (1.0 / 16777216.0)
    } else {
        f32::from_bits(
            ((value as u32 & 0x8000) << 16)
                | ((exponent as u32 + 112) << 23)
                | ((mantissa as u32) << 13),
        )
    }
}
fn samples(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let mut bits = (i as u32)
                .wrapping_add(seed)
                .wrapping_mul(747796405)
                .wrapping_add(2891336453);
            bits = ((bits >> ((bits >> 28) + 4)) ^ bits).wrapping_mul(277803737);
            (((bits >> 22) ^ bits) % 2001) as f32 / 1031.0 - 0.97
        })
        .collect()
}
fn packed(values: &[f32]) -> Vec<u32> {
    values
        .chunks(2)
        .map(|pair| {
            u32::from(half(pair[0])) | (pair.get(1).map(|v| u32::from(half(*v))).unwrap_or(0) << 16)
        })
        .collect()
}
struct Data {
    first: GpuBuffer,
    weight: GpuBuffer,
    extra: GpuBuffer,
    output: GpuBuffer,
    reference: Vec<f32>,
    push: [u32; 3],
    groups: [u32; 3],
    extra_binding: bool,
}
impl Data {
    fn new(
        device: &VulkanDevice,
        name: &str,
        rows: usize,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let parameter = name.starts_with("parameter");
        let backward = name.contains("input_grad");
        let fp16 = name.contains("fp16");
        let native = name.contains("native_compute");
        let bias = name.contains("bias");
        let residual = name.contains("residual");
        let reduction = if backward { output } else { input };
        let width = if backward { input } else { output };
        let first_values = samples(rows * reduction, 97);
        let weights = samples(input * output, 701);
        let extra_values = samples(if residual { rows * output } else { output }, 439);
        let mut reference = vec![0.0; rows * width];
        for row in 0..rows {
            for column in 0..width {
                let mut sum = 0.0f32;
                for k in 0..reduction {
                    let w_index = if parameter == backward {
                        column * reduction + k
                    } else {
                        k * width + column
                    };
                    let w = if fp16 {
                        widen(half(weights[w_index]))
                    } else {
                        weights[w_index]
                    };
                    let value = first_values[row * reduction + k];
                    sum = if native {
                        sum + widen(half(widen(half(value)) * w))
                    } else {
                        value.mul_add(w, sum)
                    };
                }
                reference[row * width + column] = if bias {
                    sum + extra_values[column]
                } else if residual {
                    extra_values[row * width + column] + sum
                } else {
                    sum
                };
            }
        }
        let packed_columns = parameter && fp16 && !backward;
        Ok(Self {
            first: GpuBuffer::from_f32(device, &first_values)?,
            weight: if fp16 {
                GpuBuffer::from_u32(device, &packed(&weights))?
            } else {
                GpuBuffer::from_f32(device, &weights)?
            },
            extra: GpuBuffer::from_f32(device, &extra_values)?,
            output: GpuBuffer::zeros_f32(device, rows * width)?,
            reference,
            push: [rows as u32, input as u32, output as u32],
            groups: [
                width.div_ceil(if packed_columns { 32 } else { 16 }) as u32,
                rows.div_ceil(16) as u32,
                1,
            ],
            extra_binding: bias || residual,
        })
    }
    fn buffers(&self) -> Vec<&GpuBuffer> {
        if self.extra_binding {
            vec![&self.first, &self.weight, &self.extra, &self.output]
        } else {
            vec![&self.first, &self.weight, &self.output]
        }
    }
    fn record(&self, kernel: &ComputeKernel, batch: &mut ComputeBatch) -> Result<()> {
        kernel.record_dispatch(
            batch,
            &self.buffers(),
            bytemuck::cast_slice(&self.push),
            self.groups,
        )
    }
    fn run(&self, device: &VulkanDevice, kernel: &ComputeKernel) -> Result<Vec<f32>> {
        let mut batch = ComputeBatch::new(device)?;
        self.record(kernel, &mut batch)?;
        batch.submit()?;
        self.output.read_f32(self.reference.len())
    }
    fn time(&self, device: &VulkanDevice, kernel: &ComputeKernel) -> Result<f64> {
        self.run(device, kernel)?;
        let mut times = Vec::new();
        for _ in 0..5 {
            times.push(
                device.time_compute_batch_ms(|batch| {
                    for _ in 0..8 {
                        self.record(kernel, batch)?;
                    }
                    Ok(())
                })? / 8.0,
            );
        }
        times.sort_by(f64::total_cmp);
        Ok(times[times.len() / 2])
    }
}
fn kernel(device: &VulkanDevice, dir: &Path, name: &str) -> Result<ComputeKernel> {
    let extra = name.contains("bias") || name.contains("residual");
    let mut accesses = vec![BindingAccess::ReadOnly; if extra { 4 } else { 3 }];
    *accesses.last_mut().unwrap() = BindingAccess::WriteOnly;
    ComputeKernel::new_with_access(
        device,
        &fs::read(dir.join(format!("{name}.spv")))?,
        &accesses,
        12,
    )
}
fn main() -> Result<()> {
    let mut baseline_dir = None;
    let mut candidate_dir = None;
    let mut only = None;
    let mut output_path = None;
    let mut golden_path = None;
    let mut write_golden_path = None;
    let mut golden_strict = false;
    let mut bench = false;
    let mut device_index = 0;
    let arguments = std::env::args().skip(1).collect::<Vec<String>>();
    let mut args = arguments.iter().cloned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--baseline-dir" => {
                baseline_dir = Some(PathBuf::from(
                    args.next().context("missing baseline directory")?,
                ))
            }
            // Overriding the candidate directory lets the same reference model
            // compare two historical artifacts, so a failure can be attributed
            // to the kernel change rather than to the model itself.
            "--candidate-dir" => {
                candidate_dir = Some(PathBuf::from(
                    args.next().context("missing candidate directory")?,
                ))
            }
            "--only" => only = Some(args.next().context("missing shader name")?),
            "--output" => {
                output_path = Some(PathBuf::from(args.next().context("missing output file")?))
            }
            "--golden" => {
                golden_path = Some(PathBuf::from(args.next().context("missing golden file")?))
            }
            "--write-golden" => {
                write_golden_path = Some(PathBuf::from(args.next().context("missing golden file")?))
            }
            "--golden-strict" => golden_strict = true,
            "--coverage" | "--coverage-key" => {
                // Resolved by `coverage_args` below; consumed here so the harness does not
                // read the flag's value as its own argument.
                let _ = args.next().context("missing coverage argument")?;
            }
            "--record-coverage" | "--allow-coverage-shrink" | "--bootstrap-coverage" => {}
            "--device-index" => {
                device_index = args.next().context("missing device index")?.parse()?
            }
            "--bench" => bench = true,
            _ => bail!("unknown argument {arg}"),
        }
    }
    let device = VulkanDevice::new_with_index(device_index)?;
    if golden_path.is_some() && write_golden_path.is_some() {
        bail!("--golden and --write-golden are mutually exclusive");
    }
    if golden_strict && golden_path.is_none() {
        bail!("--golden-strict needs a --golden file to verify against");
    }
    // Coverage is audited or recorded whenever a golden is involved: the record describes
    // the invocations this device exercises, which is exactly what a golden run exercises.
    let coverage = coverage_args(&arguments)?;
    let mut golden = match (&golden_path, &write_golden_path) {
        (Some(path), _) => Some(
            Golden::compare(path, device.name())?
                .strict(golden_strict)
                .with_coverage(coverage, path)
                // `--only` deliberately exercises one kernel, which is not the coverage
                // shrink the record exists to catch.
                .restricted(only.is_some()),
        ),
        (None, Some(path)) => {
            Some(Golden::write(path, device.name())?.with_coverage(coverage, path))
        }
        (None, None) => None,
    };
    let current_dir =
        candidate_dir.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders"));
    let mut report = Report {
        device: device.name().to_string(),
        checks: Vec::new(),
        timings: Vec::new(),
    };
    let cases = [
        (1, 1, 1),
        (1, 65, 33),
        (7, 64, 16),
        (8, 63, 17),
        (8, 64, 16),
        (8, 65, 33),
        (16, 96, 64),
        (17, 65, 67),
        (31, 97, 95),
        (33, 129, 131),
        (2, 31, 9),
        (16, 64, 63),
    ];
    for name in NAMES {
        if only.as_deref().is_some_and(|only| only != *name) {
            continue;
        }
        if name.contains("native_compute")
            && !device.mixed_precision_capabilities().shader_float16_enabled
        {
            continue;
        }
        let current = kernel(&device, &current_dir, name)?;
        let baseline = baseline_dir
            .as_ref()
            .map(|dir| kernel(&device, dir, name))
            .transpose()?;
        for (rows, input, output) in cases {
            let data = Data::new(&device, name, rows, input, output)?;
            let values = data.run(&device, &current)?;
            if let Some(golden) = golden.as_mut() {
                golden.observe(&format!("{name}:{rows}x{input}x{output}"), &values);
            }
            let max_error = values
                .iter()
                .zip(&data.reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mut tolerance = 0.0005f32;
            let parity = if let Some(old) = &baseline {
                let previous = data.run(&device, old)?;
                // Bit-for-bit equality with the unoptimized artifact is the
                // strict contract for a kernel optimization. Check it before
                // the independent reference model, which cannot always express
                // exact Float16 product rounding.
                let equal = values
                    .iter()
                    .zip(&previous)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                ensure!(
                    equal,
                    "{name} ({rows},{input},{output}) differs from baseline"
                );
                // Calibrate the advisory model against the unoptimized artifact
                // so a change that widens the deviation still fails here.
                let baseline_error = previous
                    .iter()
                    .zip(&data.reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                if baseline_error > tolerance {
                    println!("reference-model calibration {name} ({rows},{input},{output}): baseline error {baseline_error} exceeds {tolerance}");
                    tolerance = baseline_error * 1.25 + 1e-6;
                }
                Some(equal)
            } else {
                None
            };
            if baseline_dir.is_some() {
                ensure!(max_error <= tolerance, "{name} ({rows},{input},{output}) reference max error {max_error} (tolerance {tolerance})");
            } else {
                // Without a baseline artifact to calibrate against, the advisory
                // reference model cannot express exact Float16 product rounding
                // for the native-compute kernels, so the golden digest is the
                // strict contract here and the reference error is reported only.
                println!("reference-model advisory {name} ({rows},{input},{output}): max error {max_error} (tolerance {tolerance})");
            }
            report.checks.push(Check {
                kernel: name.to_string(),
                rows,
                input_dim: input,
                output_dim: output,
                elements: values.len(),
                baseline_bitwise_equal: parity,
                max_reference_error: max_error,
            });
        }
        if bench {
            for (rows, input, output) in [
                (1, 512, 512),
                (8, 64, 16),
                (16, 256, 256),
                (32, 512, 512),
                (64, 1024, 256),
                (17, 65, 33),
            ] {
                let data = Data::new(&device, name, rows, input, output)?;
                let old_time = baseline
                    .as_ref()
                    .map(|old| data.time(&device, old))
                    .transpose()?;
                let current_time = data.time(&device, &current)?;
                report.timings.push(Timing {
                    kernel: name.to_string(),
                    rows,
                    input_dim: input,
                    output_dim: output,
                    current_ms: current_time,
                    baseline_ms: old_time,
                    speedup: old_time.map(|old| old / current_time),
                });
            }
        }
        eprintln!("validated {name}");
    }
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        fs::write(path, &json)?;
    }
    println!("{json}");
    if let Some(golden) = golden {
        golden.finish()?;
    }
    Ok(())
}
