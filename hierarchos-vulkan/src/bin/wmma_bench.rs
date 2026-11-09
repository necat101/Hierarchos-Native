//! Paired ABBA benchmark for the opt-in WMMA arms against the existing FP32
//! kernel and the plain FP16-storage control.
//!
//! Every arm computes the *same* logical GEMM with the *same* layout, so the
//! only difference is the arithmetic unit and the operand precision. Rounds
//! alternate forward/reverse order (ABBA), each round takes a fresh block of
//! timed iterations, and the report carries per-round ratios plus the paired
//! median, min and max. A kernel-level win is reported as GEMM call latency; no
//! end-to-end training-step claim is derived from it.
//!
//! Usage: `hierarchos-vulkan-wmma-bench [--wmma-mode mixed|full-fp16]
//!        [--rounds N] [--iterations N] [--json]`
//!
//! `--wmma-mode off` measures the FP32 baseline and the plain FP16 control only.

use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::wmma::{
    drift, reference_fp32_gemm, GemmLayout, WmmaGemm, WmmaGemmArm, WmmaMode,
};
use hierarchos_vulkan::VulkanDevice;

/// Representative dense-linear/attention-projection geometries. The last two
/// are small shapes where cooperative tiling is expected to lose.
const SHAPES: [(usize, usize, usize, &str); 6] = [
    (512, 768, 768, "attention q/k/v projection"),
    (512, 2048, 768, "ffn up projection"),
    (2048, 768, 2048, "ffn down projection"),
    (128, 8192, 768, "vocabulary projection"),
    (64, 64, 64, "lora/small control"),
    (16, 128, 32, "tiny control"),
];

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.is_empty() {
        f64::NAN
    } else if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn deterministic(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((state >> 8) & 0xFFFF) as f32 / 65_535.0;
            (unit * 2.0 - 1.0) * (0.05 + (index % 5) as f32 * 0.01)
        })
        .collect()
}

struct Measurement {
    label: &'static str,
    rounds: Vec<f64>,
    drift_vs_fp32: Option<f64>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("wmma-bench: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut mode = WmmaMode::from_env()?;
    let mut rounds = 5usize;
    let mut iterations = 40usize;
    let mut json = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wmma-mode" => {
                let value = args.next().context("--wmma-mode requires a value")?;
                mode = WmmaMode::parse(&value)?;
            }
            "--rounds" => {
                rounds = args
                    .next()
                    .context("--rounds requires a value")?
                    .parse()
                    .context("--rounds must be an integer")?;
            }
            "--iterations" => {
                iterations = args
                    .next()
                    .context("--iterations requires a value")?
                    .parse()
                    .context("--iterations must be an integer")?;
            }
            "--json" => json = true,
            "--help" | "-h" => {
                println!(
                    "hierarchos-vulkan-wmma-bench [--wmma-mode off|mixed|full-fp16] \
                     [--rounds N] [--iterations N] [--json]"
                );
                return Ok(());
            }
            other => bail!("unsupported argument {other:?}"),
        }
    }
    if rounds == 0 || iterations == 0 {
        bail!("--rounds and --iterations must be positive");
    }

    let device = VulkanDevice::new_for_wmma(mode)?;
    let capabilities = device.wmma_capabilities();
    capabilities.require(mode)?;
    let gemm = WmmaGemm::new(&device, mode)?;

    let mut arms: Vec<WmmaGemmArm> = vec![WmmaGemmArm::Fp32Baseline, WmmaGemmArm::PlainF16Storage];
    match mode {
        WmmaMode::Mixed => arms.push(WmmaGemmArm::MixedFp32Accumulate),
        WmmaMode::FullFp16 => arms.push(WmmaGemmArm::FullFp16Accumulate),
        WmmaMode::Off => {}
    }

    // The FP32 kernel stores B as [columns, reduction].
    let layout = GemmLayout {
        transposed_a: false,
        transposed_b: true,
    };

    if !json {
        println!("WMMA paired ABBA benchmark");
        println!("device: {}", device.name());
        println!("mode: {} ({})", mode.label(), mode.description());
        print!("{}", capabilities.diagnostics());
        println!(
            "rounds={rounds} iterations/round={iterations} layout=A[rows,k]xB^T[columns,k]"
        );
    }

    let mut json_blocks: Vec<String> = Vec::new();

    for (rows, columns, reduction, label) in SHAPES {
        let a = deterministic(rows * reduction, 0x1234_5678 ^ rows as u32);
        let b = deterministic(columns * reduction, 0x9E37_79B9 ^ columns as u32);
        let reference = reference_fp32_gemm(&a, &b, rows, columns, reduction, layout)?;

        let mut measurements: Vec<Measurement> = Vec::new();
        // Warm every arm once so the first round is not skewed by pipeline
        // creation or allocation of the shared buffers.
        for arm in &arms {
            let _ = gemm.bench_repeated(*arm, &a, &b, rows, columns, reduction, layout, 3)?;
        }
        for round in 0..rounds {
            let order: Vec<&WmmaGemmArm> = if round % 2 == 0 {
                arms.iter().collect()
            } else {
                arms.iter().rev().collect()
            };
            for arm in order {
                let samples = gemm.bench_repeated(
                    *arm, &a, &b, rows, columns, reduction, layout, iterations,
                )?;
                let round_median = median(&samples);
                match measurements.iter_mut().find(|entry| entry.label == arm.label()) {
                    Some(entry) => entry.rounds.push(round_median),
                    None => measurements.push(Measurement {
                        label: arm.label(),
                        rounds: vec![round_median],
                        drift_vs_fp32: None,
                    }),
                }
            }
        }

        // One correctness sample per arm against the f64 FP32 reference, so a
        // speed number is never reported without its numerical cost.
        for entry in measurements.iter_mut() {
            let arm = arms
                .iter()
                .copied()
                .find(|arm| arm.label() == entry.label)
                .expect("measurement arm must be known");
            let observed = gemm.matmul(arm, &a, &b, rows, columns, reduction, layout)?;
            entry.drift_vs_fp32 = Some(drift(&reference, &observed)?.max_abs);
        }

        let baseline = measurements
            .iter()
            .find(|entry| entry.label == WmmaGemmArm::Fp32Baseline.label())
            .expect("baseline arm is always measured");
        let baseline_median = median(&baseline.rounds);

        if json {
            let entries: Vec<String> = measurements
                .iter()
                .map(|entry| {
                    let paired: Vec<f64> = entry
                        .rounds
                        .iter()
                        .zip(&baseline.rounds)
                        .map(|(arm, base)| base / arm)
                        .collect();
                    format!(
                        "      {{\"arm\":{:?},\"median_ms\":{:.6},\"min_ms\":{:.6},\"max_ms\":{:.6},\"speedup_vs_fp32\":{:.4},\"drift_vs_fp32_max_abs\":{:.6e}}}",
                        entry.label,
                        median(&entry.rounds) * 1.0e3,
                        entry.rounds.iter().copied().fold(f64::INFINITY, f64::min) * 1.0e3,
                        entry.rounds.iter().copied().fold(0.0f64, f64::max) * 1.0e3,
                        median(&paired),
                        entry.drift_vs_fp32.unwrap_or(f64::NAN)
                    )
                })
                .collect();
            json_blocks.push(format!(
                "    {{\"shape\":\"{rows}x{columns}x{reduction}\",\"label\":{label:?},\"arms\":[\n{}\n    ]}}",
                entries.join(",\n")
            ));
        } else {
            println!("\nshape {rows}x{columns}x{reduction} ({label})");
            println!(
                "  {:<26} {:>10} {:>10} {:>10} {:>8} {:>12}",
                "arm", "median ms", "min ms", "max ms", "speedup", "drift_max"
            );
            for entry in &measurements {
                let paired: Vec<f64> = entry
                    .rounds
                    .iter()
                    .zip(&baseline.rounds)
                    .map(|(arm, base)| base / arm)
                    .collect();
                println!(
                    "  {:<26} {:>10.4} {:>10.4} {:>10.4} {:>7.3}x {:>12.3e}",
                    entry.label,
                    median(&entry.rounds) * 1.0e3,
                    entry.rounds.iter().copied().fold(f64::INFINITY, f64::min) * 1.0e3,
                    entry.rounds.iter().copied().fold(0.0f64, f64::max) * 1.0e3,
                    median(&paired),
                    entry.drift_vs_fp32.unwrap_or(f64::NAN)
                );
            }
            let scalable = measurements
                .iter()
                .filter(|entry| entry.label != WmmaGemmArm::Fp32Baseline.label())
                .map(|entry| {
                    let paired: Vec<f64> = entry
                        .rounds
                        .iter()
                        .zip(&baseline.rounds)
                        .map(|(arm, base)| base / arm)
                        .collect();
                    format!("{}={:.3}x (all rounds {})", entry.label, median(&paired),
                        if paired.iter().all(|ratio| *ratio > 1.0) {
                            ">1"
                        } else if paired.iter().all(|ratio| *ratio < 1.0) {
                            "<1"
                        } else {
                            "mixed"
                        })
                })
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "  baseline median {:.4} ms; {scalable}",
                baseline_median * 1.0e3
            );
        }
    }

    if json {
        let counts = gemm.dispatch_counts();
        println!(
            "{{\n  \"device\": {:?},\n  \"mode\": {:?},\n  \"rounds\": {},\n  \"iterations_per_round\": {},\n  \"cooperative_dispatches\": {},\n  \"plain_dispatches\": {},\n  \"shape_fallbacks\": {},\n  \"shapes\": [\n{}\n  ]\n}}",
            device.name(),
            mode.label(),
            rounds,
            iterations,
            counts.cooperative,
            counts.plain_f16,
            counts.shape_fallbacks,
            json_blocks.join(",\n")
        );
    } else {
        let counts = gemm.dispatch_counts();
        println!(
            "\ndispatch accounting: cooperative={} plain_f16={} shape_fallbacks={}",
            counts.cooperative, counts.plain_f16, counts.shape_fallbacks
        );
    }
    Ok(())
}
