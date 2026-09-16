use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{RwkvLowRankOp, RwkvLowRankParameterGradArithmetic, VulkanDevice};
use serde::Serialize;

const DEFAULT_GEOMETRIES: &[(usize, usize, usize)] =
    &[(1, 32, 128), (4, 64, 256), (8, 96, 448), (16, 128, 448)];

const ALL_MODES: &[RwkvLowRankParameterGradArithmetic] = &[
    RwkvLowRankParameterGradArithmetic::Fp32,
    RwkvLowRankParameterGradArithmetic::NativeFp16,
    RwkvLowRankParameterGradArithmetic::NativeFp16WidenedProduct,
    RwkvLowRankParameterGradArithmetic::NativeFp16CompensatedOperands,
];

#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    input_dim: usize,
    output_dim: usize,
}

struct Args {
    geometries: Vec<Geometry>,
    modes: Vec<RwkvLowRankParameterGradArithmetic>,
    warmup_iterations: usize,
    measured_iterations: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args_os().skip(1);
        let mut geometries = Vec::new();
        let mut modes = None;
        let mut warmup_iterations = 8usize;
        let mut measured_iterations = 64usize;

        while let Some(arg) = args.next() {
            match arg.to_string_lossy().as_ref() {
                "--geometry" => {
                    let raw = args
                        .next()
                        .context("--geometry requires ROWSxINPUTxOUTPUT")?;
                    geometries.push(parse_geometry(&raw.to_string_lossy())?);
                }
                "--modes" => {
                    let raw = args
                        .next()
                        .context("--modes requires a comma-separated value")?;
                    modes = Some(parse_modes(&raw.to_string_lossy())?);
                }
                "--warmup" => {
                    warmup_iterations = parse_usize(args.next(), "--warmup", true)?;
                }
                "--iterations" => {
                    measured_iterations = parse_usize(args.next(), "--iterations", false)?;
                }
                "--help" | "-h" => {
                    println!(
                        "usage: hierarchos-vulkan-rwkv-low-rank-dw-bench \
                         [--geometry ROWSxINPUTxOUTPUT]... \
                         [--modes all|fp32,native-fp16,native-fp16-widened-product,native-fp16-compensated-operands] \
                         [--warmup N] [--iterations N]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other:?}"),
            }
        }

        if geometries.is_empty() {
            geometries = DEFAULT_GEOMETRIES
                .iter()
                .map(|&(rows, input_dim, output_dim)| Geometry {
                    rows,
                    input_dim,
                    output_dim,
                })
                .collect();
        }
        Ok(Self {
            geometries,
            modes: modes.unwrap_or_else(|| ALL_MODES.to_vec()),
            warmup_iterations,
            measured_iterations,
        })
    }
}

#[derive(Serialize)]
struct Output {
    device: String,
    warmup_iterations: usize,
    measured_iterations: usize,
    kernel_profile_hint: &'static str,
    records: Vec<Record>,
}

#[derive(Serialize)]
struct Record {
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    arithmetic: String,
    available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    elapsed_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dispatches_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gmacs_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_fp16_gproducts_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kernel_resident_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allocator_live_buffer_bytes_delta: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allocator_reserved_bytes_delta: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allocator_driver_allocation_count_delta: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_abs_diff_vs_fp32: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rms_diff_vs_fp32: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_abs_fp32_reference: Option<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let max_rows = args
        .geometries
        .iter()
        .map(|geometry| geometry.rows)
        .max()
        .context("dW benchmark requires at least one geometry")?;
    let max_rank = args
        .geometries
        .iter()
        .map(|geometry| geometry.input_dim)
        .max()
        .context("dW benchmark requires at least one geometry")?;
    let max_width = args
        .geometries
        .iter()
        .map(|geometry| geometry.output_dim)
        .max()
        .context("dW benchmark requires at least one geometry")?;

    let device = VulkanDevice::new()?;
    let device_name = device.name().to_string();
    let zeros_width = vec![0.0f32; max_width];
    let first_stage = deterministic_values(max_width * max_rank, 0.001, 3);
    let second_stage = deterministic_values(max_rank * max_width, 0.001, 7);
    let op = RwkvLowRankOp::new(
        device,
        max_width,
        max_rank,
        max_rank,
        max_rank,
        max_rows,
        &zeros_width,
        &zeros_width,
        &zeros_width,
        &zeros_width,
        &first_stage,
        &second_stage,
        &zeros_width,
        &first_stage,
        &second_stage,
        &first_stage,
        &second_stage,
    )?;

    let mut records = Vec::new();
    for geometry in &args.geometries {
        let input = deterministic_values(geometry.rows * geometry.input_dim, 0.013, 11);
        let grad_output = deterministic_values(geometry.rows * geometry.output_dim, 0.009, 19);
        let fp32_reference = op.diagnose_weight_grad(
            RwkvLowRankParameterGradArithmetic::Fp32,
            geometry.rows,
            geometry.input_dim,
            geometry.output_dim,
            &input,
            &grad_output,
        )?;
        let max_abs_fp32_reference = fp32_reference
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);

        for &mode in &args.modes {
            match op.benchmark_weight_grad(
                mode,
                geometry.rows,
                geometry.input_dim,
                geometry.output_dim,
                &input,
                &grad_output,
                args.warmup_iterations,
                args.measured_iterations,
            ) {
                Ok(benchmark) => {
                    let (max_abs_diff, rms_diff) =
                        difference_stats(&benchmark.gradient, &fp32_reference)?;
                    records.push(Record {
                        rows: geometry.rows,
                        input_dim: geometry.input_dim,
                        output_dim: geometry.output_dim,
                        arithmetic: mode.label().to_string(),
                        available: true,
                        error: None,
                        elapsed_ms: Some(benchmark.elapsed_seconds * 1_000.0),
                        dispatches_per_second: Some(benchmark.dispatches_per_second),
                        gmacs_per_second: Some(benchmark.macs_per_second / 1.0e9),
                        native_fp16_gproducts_per_second: Some(
                            benchmark.native_fp16_products_per_second / 1.0e9,
                        ),
                        kernel_resident_bytes: Some(benchmark.kernel_resident_bytes),
                        allocator_live_buffer_bytes_delta: Some(
                            benchmark.allocator_live_buffer_bytes_delta,
                        ),
                        allocator_reserved_bytes_delta: Some(
                            benchmark.allocator_reserved_bytes_delta,
                        ),
                        allocator_driver_allocation_count_delta: Some(
                            benchmark.allocator_driver_allocation_count_delta,
                        ),
                        max_abs_diff_vs_fp32: Some(max_abs_diff),
                        rms_diff_vs_fp32: Some(rms_diff),
                        max_abs_fp32_reference: Some(max_abs_fp32_reference),
                    });
                }
                Err(error) => records.push(Record {
                    rows: geometry.rows,
                    input_dim: geometry.input_dim,
                    output_dim: geometry.output_dim,
                    arithmetic: mode.label().to_string(),
                    available: false,
                    error: Some(format!("{error:#}")),
                    elapsed_ms: None,
                    dispatches_per_second: None,
                    gmacs_per_second: None,
                    native_fp16_gproducts_per_second: None,
                    kernel_resident_bytes: None,
                    allocator_live_buffer_bytes_delta: None,
                    allocator_reserved_bytes_delta: None,
                    allocator_driver_allocation_count_delta: None,
                    max_abs_diff_vs_fp32: None,
                    rms_diff_vs_fp32: None,
                    max_abs_fp32_reference: Some(max_abs_fp32_reference),
                }),
            }
        }
    }

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: device_name,
            warmup_iterations: args.warmup_iterations,
            measured_iterations: args.measured_iterations,
            kernel_profile_hint: "set HIERARCHOS_VULKAN_PROFILE_KERNELS=1 for Vulkan timestamp-query output on stderr",
            records,
        })?
    );
    Ok(())
}

fn parse_geometry(raw: &str) -> Result<Geometry> {
    let normalized = raw.replace(['X', '*'], "x");
    let parts = normalized.split('x').collect::<Vec<_>>();
    if parts.len() != 3 {
        bail!("--geometry must be ROWSxINPUTxOUTPUT; got {raw:?}");
    }
    let parse = |value: &str, field: &str| -> Result<usize> {
        let parsed = value
            .parse::<usize>()
            .with_context(|| format!("invalid {field} in --geometry {raw:?}"))?;
        if parsed == 0 {
            bail!("{field} in --geometry must be positive");
        }
        Ok(parsed)
    };
    Ok(Geometry {
        rows: parse(parts[0], "rows")?,
        input_dim: parse(parts[1], "input_dim")?,
        output_dim: parse(parts[2], "output_dim")?,
    })
}

fn parse_modes(raw: &str) -> Result<Vec<RwkvLowRankParameterGradArithmetic>> {
    if raw.eq_ignore_ascii_case("all") {
        return Ok(ALL_MODES.to_vec());
    }
    let mut modes = Vec::new();
    for value in raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let mode = match value {
            "fp32" => RwkvLowRankParameterGradArithmetic::Fp32,
            "native" | "native-fp16" => RwkvLowRankParameterGradArithmetic::NativeFp16,
            "widened" | "native-fp16-widened-product" => {
                RwkvLowRankParameterGradArithmetic::NativeFp16WidenedProduct
            }
            "compensated" | "native-fp16-compensated-operands" => {
                RwkvLowRankParameterGradArithmetic::NativeFp16CompensatedOperands
            }
            _ => bail!("unknown dW arithmetic mode {value:?}"),
        };
        if !modes.contains(&mode) {
            modes.push(mode);
        }
    }
    if modes.is_empty() {
        bail!("--modes must select at least one dW arithmetic mode");
    }
    Ok(modes)
}

fn parse_usize(value: Option<std::ffi::OsString>, name: &str, allow_zero: bool) -> Result<usize> {
    let raw = value.with_context(|| format!("{name} requires a value"))?;
    let parsed = raw
        .to_string_lossy()
        .parse::<usize>()
        .with_context(|| format!("{name} must be an integer"))?;
    if !allow_zero && parsed == 0 {
        bail!("{name} must be positive");
    }
    Ok(parsed)
}

fn deterministic_values(len: usize, scale: f32, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let centered = ((index * 37 + salt * 17) % 211) as f32 - 105.0;
            centered * scale
        })
        .collect()
}

fn difference_stats(actual: &[f32], reference: &[f32]) -> Result<(f32, f32)> {
    if actual.len() != reference.len() {
        bail!(
            "dW benchmark result length mismatch: actual={} reference={}",
            actual.len(),
            reference.len()
        );
    }
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let diff = (actual - reference).abs();
        max_abs = max_abs.max(diff);
        sum_sq += f64::from(diff) * f64::from(diff);
    }
    let rms = if actual.is_empty() {
        0.0
    } else {
        (sum_sq / actual.len() as f64).sqrt() as f32
    };
    Ok((max_abs, rms))
}
