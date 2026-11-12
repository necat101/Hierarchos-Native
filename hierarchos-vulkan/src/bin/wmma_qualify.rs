//! Numerical qualification receipts for the opt-in WMMA policies.
//!
//! Every comparison is kept separate, exactly as the precision contracts
//! require:
//!
//! 1. FP32 CPU reference (f64 accumulation) — the baseline.
//! 2. WMMA mixed versus that FP32 reference (quantization drift).
//! 3. WMMA mixed versus its matched FP16-operand/FP32-accumulate oracle
//!    (implementation correctness).
//! 4. WMMA full-FP16 versus FP32.
//! 5. WMMA full-FP16 versus its matched FP16-accumulate oracle.
//! 6. Plain FP16-storage GEMM (no cooperative matrices) versus the same
//!    matched oracle, so the ablation control is qualified too.
//!
//! The FP32 `2e-7` gate is *not* applied here: FP16 operand quantization cannot
//! meet it, and pretending otherwise would either weaken the real FP32 gate or
//! fail every reduced-precision mode for the wrong reason.
//!
//! Exit status is non-zero when any gate fails or when a requested mode is not
//! supported by the device, so a receipt cannot be mistaken for a pass.

use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::wmma::{
    drift, f16_ulp, logit_agreement, oracle_gemm, reference_fp32_gemm, GemmLayout,
    OracleAccumulation, WmmaDrift, WmmaGemm, WmmaGemmArm, WmmaMode, FULL_VS_FP32_MAX_ABS,
    FULL_VS_FP32_RMS, FULL_VS_MATCHED_ORACLE_MAX_ABS, MIXED_VS_FP32_MAX_ABS, MIXED_VS_FP32_RMS,
    MIXED_VS_MATCHED_ORACLE_MAX_ABS,
};

/// Matched-oracle tolerance for an FP16-accumulating arm.
///
/// FP16 accumulation has no single absolute error bound: the error scales with
/// the accumulator magnitude. Two binary16 ULPs at the largest reference
/// magnitude is the documented semantic limit, floored at the mixed-arm target
/// so a tiny-magnitude fixture cannot demand zero error.
fn f16_matched_limit(reference: &[f32]) -> f64 {
    let magnitude = reference
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    (2.0 * f64::from(f16_ulp(magnitude))).max(FULL_VS_MATCHED_ORACLE_MAX_ABS)
}
use hierarchos_vulkan::VulkanDevice;

/// Small deterministic fixtures. These are deliberately tiny: the point is a
/// reproducible kernel/optimizer contract, not a model-scale benchmark.
const FIXTURES: [(usize, usize, usize); 6] = [
    (64, 48, 40),   // ordinary tile-aligned-ish shape
    (33, 17, 19),   // irregular, partial tiles on both axes
    (128, 128, 96), // several k slices
    (16, 1024, 64), // minimum rows, wide output
    (256, 64, 16),  // single k slice
    (37, 8192, 48), // vocabulary-scale output
];

fn usage() -> &'static str {
    "hierarchos-vulkan-wmma-qualify [--wmma-mode off|mixed|full-fp16] [--seed N] [--json]"
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("wmma-qualify: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn deterministic_matrix(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((state >> 8) & 0xFFFF) as f32 / 65_535.0;
            (unit * 2.0 - 1.0) * (1.0 + (index % 7) as f32 * 0.15)
        })
        .collect()
}

struct Gate {
    label: String,
    measured: WmmaDrift,
    max_abs_limit: f64,
    rms_limit: Option<f64>,
    top1: Option<(usize, usize, usize)>,
    mean_top5: Option<f64>,
    max_softmax: Option<f64>,
}

impl Gate {
    fn passed(&self) -> bool {
        self.measured.all_finite()
            && self.measured.max_abs <= self.max_abs_limit
            && self.rms_limit.is_none_or(|limit| self.measured.rms <= limit)
    }

    fn render(&self) -> String {
        let mut line = format!(
            "    {:<38} max_abs={:.6e} rms={:.6e} rel={:.6e} nonfinite={} limit_max={:.1e}{}",
            self.label,
            self.measured.max_abs,
            self.measured.rms,
            self.measured.scaled_relative(),
            self.measured.nonfinite,
            self.max_abs_limit,
            match self.rms_limit {
                Some(limit) => format!(" limit_rms={limit:.1e}"),
                None => String::new(),
            }
        );
        if let Some((top1, rows, near_ties)) = self.top1 {
            line.push_str(&format!(
                " top1={top1}/{rows} near_ties={near_ties}"
            ));
        }
        if let Some(overlap) = self.mean_top5 {
            line.push_str(&format!(" top5_overlap={overlap:.4}"));
        }
        if let Some(softmax) = self.max_softmax {
            line.push_str(&format!(" softmax_max_abs={softmax:.3e}"));
        }
        line.push_str(if self.passed() { "  PASS" } else { "  FAIL" });
        line
    }
}

fn main_gate(label: &str, reference: &[f32], actual: &[f32], max_abs: f64, rms: Option<f64>) -> Result<Gate> {
    Ok(Gate {
        label: label.to_owned(),
        measured: drift(reference, actual)?,
        max_abs_limit: max_abs,
        rms_limit: rms,
        top1: None,
        mean_top5: None,
        max_softmax: None,
    })
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut mode = WmmaMode::from_env()?;
    let mut seed = 0x5EED_1234u32;
    let mut json = false;
    let mut dump = None;
    let mut samples = Vec::new();
    let mut full_accumulation = OracleAccumulation::Fp16TileK;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wmma-mode" => {
                let value = args.next().context("--wmma-mode requires a value")?;
                mode = WmmaMode::parse(&value)?;
            }
            "--seed" => {
                let value = args.next().context("--seed requires a value")?;
                seed = value.parse().context("--seed must be an integer")?;
            }
            "--json" => json = true,
            "--full-oracle" => full_accumulation = match args.next().as_deref() {
                Some("tile16") => OracleAccumulation::Fp16TileK,
                Some("rdna3-dot2") => OracleAccumulation::Fp16Dot2,
                _ => bail!("--full-oracle requires tile16 or rdna3-dot2"),
            },
            "--dump-operands" => dump = Some(std::path::PathBuf::from(
                args.next().context("--dump-operands requires a new output path")?)),
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => bail!("unsupported argument {other:?}\n\n{}", usage()),
        }
    }
    if mode == WmmaMode::Off {
        bail!(
            "qualification needs an accelerated policy; pass --wmma-mode mixed or full-fp16 \
             (the FP32 baseline gate is the existing native suite and is not redefined here)"
        );
    }

    let device = VulkanDevice::new_for_wmma(mode)?;
    let capabilities = device.wmma_capabilities();
    capabilities.require(mode)?;
    let gemm = WmmaGemm::new(&device, mode)?;

    let mut gates: Vec<Gate> = Vec::new();
    let mut layout_gates: Vec<Gate> = Vec::new();

    for (index, (rows, columns, reduction)) in FIXTURES.iter().copied().enumerate() {
        let a = deterministic_matrix(rows * reduction, seed ^ (index as u32 + 1));
        let b = deterministic_matrix(reduction * columns, seed ^ (index as u32 + 101));

        let fp32 = reference_fp32_gemm(&a, &b, rows, columns, reduction, GemmLayout::default())?;
        let mixed_oracle = oracle_gemm(
            &a,
            &b,
            rows,
            columns,
            reduction,
            GemmLayout::default(),
            OracleAccumulation::Fp32,
        )?;
        let full_oracle = oracle_gemm(
            &a,
            &b,
            rows,
            columns,
            reduction,
            GemmLayout::default(),
            full_accumulation,
        )?;
        let full_match_limit = f16_matched_limit(&full_oracle);

        let fixture = format!("{rows}x{columns}x{reduction}");
        match mode {
            WmmaMode::Mixed => {
                let observed = gemm.matmul(
                    WmmaGemmArm::MixedFp32Accumulate,
                    &a,
                    &b,
                    rows,
                    columns,
                    reduction,
                    GemmLayout::default(),
                )?;
                let mut vs_fp32 = main_gate(
                    &format!("{fixture} wmma-mixed vs FP32"),
                    &fp32,
                    &observed,
                    MIXED_VS_FP32_MAX_ABS,
                    Some(MIXED_VS_FP32_RMS),
                )?;
                let agreement = logit_agreement(&fp32, &observed, rows, columns)?;
                vs_fp32.top1 = Some((
                    agreement.top1_matches,
                    agreement.rows,
                    agreement.near_tie_top1,
                ));
                vs_fp32.mean_top5 = Some(agreement.mean_top5_overlap);
                vs_fp32.max_softmax = Some(agreement.max_softmax_abs);
                gates.push(vs_fp32);
                gates.push(main_gate(
                    &format!("{fixture} wmma-mixed vs matched oracle"),
                    &mixed_oracle,
                    &observed,
                    MIXED_VS_MATCHED_ORACLE_MAX_ABS,
                    None,
                )?);
                // The matched oracle itself is checked against FP32 so the
                // quantization budget is attributed, not assumed.
                gates.push(main_gate(
                    &format!("{fixture} matched oracle vs FP32"),
                    &fp32,
                    &mixed_oracle,
                    MIXED_VS_FP32_MAX_ABS,
                    Some(MIXED_VS_FP32_RMS),
                )?);
            }
            WmmaMode::FullFp16 => {
                let observed = gemm.matmul(
                    WmmaGemmArm::FullFp16Accumulate,
                    &a,
                    &b,
                    rows,
                    columns,
                    reduction,
                    GemmLayout::default(),
                )?;
                if dump.is_some() {
                    samples.push(serde_json::json!({"m":rows,"n":columns,"k":reduction,
                        "trans_a":false,"trans_b":false,"a":a,"b":b,"actual":observed}));
                }
                let mut vs_fp32 = main_gate(
                    &format!("{fixture} wmma-full-fp16 vs FP32"),
                    &fp32,
                    &observed,
                    FULL_VS_FP32_MAX_ABS,
                    Some(FULL_VS_FP32_RMS),
                )?;
                let agreement = logit_agreement(&fp32, &observed, rows, columns)?;
                vs_fp32.top1 = Some((
                    agreement.top1_matches,
                    agreement.rows,
                    agreement.near_tie_top1,
                ));
                vs_fp32.mean_top5 = Some(agreement.mean_top5_overlap);
                vs_fp32.max_softmax = Some(agreement.max_softmax_abs);
                gates.push(vs_fp32);
                gates.push(main_gate(
                    &format!("{fixture} wmma-full-fp16 vs matched oracle"),
                    &full_oracle,
                    &observed,
                    full_match_limit,
                    None,
                )?);
                gates.push(main_gate(
                    &format!("{fixture} matched fp16 oracle vs FP32"),
                    &fp32,
                    &full_oracle,
                    FULL_VS_FP32_MAX_ABS,
                    Some(FULL_VS_FP32_RMS),
                )?);
            }
            WmmaMode::Off => unreachable!(),
        }

        // Layout coverage: transpose flags must not change which operand is
        // which. Only the small fixtures are exercised in all four layouts to
        // keep the f64 reference affordable.
        if index < 3 {
            for layout in [
                GemmLayout {
                    transposed_a: true,
                    transposed_b: false,
                },
                GemmLayout {
                    transposed_a: false,
                    transposed_b: true,
                },
                GemmLayout {
                    transposed_a: true,
                    transposed_b: true,
                },
            ] {
                let fp32_layout = reference_fp32_gemm(&a, &b, rows, columns, reduction, layout)?;
                let (arm, accumulation) = match mode {
                    WmmaMode::Mixed => (WmmaGemmArm::MixedFp32Accumulate, OracleAccumulation::Fp32),
                    WmmaMode::FullFp16 => {
                        (WmmaGemmArm::FullFp16Accumulate, full_accumulation)
                    }
                    WmmaMode::Off => unreachable!(),
                };
                let oracle =
                    oracle_gemm(&a, &b, rows, columns, reduction, layout, accumulation)?;
                let observed = gemm.matmul(arm, &a, &b, rows, columns, reduction, layout)?;
                if dump.is_some() {
                    samples.push(serde_json::json!({"m":rows,"n":columns,"k":reduction,
                        "trans_a":layout.transposed_a,"trans_b":layout.transposed_b,
                        "a":a,"b":b,"actual":observed}));
                }
                layout_gates.push(main_gate(
                    &format!(
                        "{fixture} trans_a={} trans_b={} vs oracle",
                        layout.transposed_a, layout.transposed_b
                    ),
                    &oracle,
                    &observed,
                    if accumulation == OracleAccumulation::Fp32 {
                        MIXED_VS_MATCHED_ORACLE_MAX_ABS
                    } else {
                        f16_matched_limit(&oracle)
                    },
                    None,
                )?);
                let mut vs_fp32 = main_gate(
                    &format!(
                        "{fixture} trans_a={} trans_b={} vs FP32",
                        layout.transposed_a, layout.transposed_b
                    ),
                    &fp32_layout,
                    &observed,
                    if accumulation == OracleAccumulation::Fp32 {
                        MIXED_VS_FP32_MAX_ABS
                    } else {
                        FULL_VS_FP32_MAX_ABS
                    },
                    Some(if accumulation == OracleAccumulation::Fp32 {
                        MIXED_VS_FP32_RMS
                    } else {
                        FULL_VS_FP32_RMS
                    }),
                )?;
                vs_fp32.max_softmax = None;
                layout_gates.push(vs_fp32);
            }
        }
    }

    if let Some(path) = dump {
        let file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        serde_json::to_writer(file, &samples)?;
    }

    // Plain FP16-storage control: the same operand precision without matrix
    // cores. It must meet the matched-oracle bound, otherwise an ablation could
    // not attribute a regression to WMMA itself.
    let control_a = deterministic_matrix(64 * 64, seed ^ 0xAA);
    let control_b = deterministic_matrix(64 * 64, seed ^ 0xBB);
    let control_oracle = oracle_gemm(
        &control_a,
        &control_b,
        64,
        64,
        64,
        GemmLayout::default(),
        OracleAccumulation::Fp32,
    )?;
    let control = gemm.matmul(
        WmmaGemmArm::PlainF16Storage,
        &control_a,
        &control_b,
        64,
        64,
        64,
        GemmLayout::default(),
    )?;
    let control_gate = main_gate(
        "64x64x64 plain-f16-storage vs matched oracle",
        &control_oracle,
        &control,
        MIXED_VS_MATCHED_ORACLE_MAX_ABS,
        None,
    )?;

    let all: Vec<&Gate> = gates.iter().chain(&layout_gates).chain([&control_gate]).collect();
    let failed = all.iter().filter(|gate| !gate.passed()).count();
    let counts = gemm.dispatch_counts();
    // Fixtures that fail any gate are reported as unqualified rather than
    // summarised away: a mode can be qualified on some geometries and not others.
    let mut unqualified: Vec<&str> = all
        .iter()
        .filter(|gate| !gate.passed())
        .map(|gate| gate.label.as_str())
        .collect();
    unqualified.sort_unstable();
    unqualified.dedup();

    if json {
        let rows: Vec<String> = all
            .iter()
            .map(|gate| {
                format!(
                    "    {{\"label\":{:?},\"max_abs\":{:.9e},\"rms\":{:.9e},\"relative\":{:.9e},\"nonfinite\":{},\"max_abs_limit\":{:.3e},\"rms_limit\":{},\"passed\":{}}}",
                    gate.label,
                    gate.measured.max_abs,
                    gate.measured.rms,
                    gate.measured.scaled_relative(),
                    gate.measured.nonfinite,
                    gate.max_abs_limit,
                    gate.rms_limit.map(|limit| format!("{limit:.3e}")).unwrap_or_else(|| "null".to_owned()),
                    gate.passed()
                )
            })
            .collect();
        println!(
            "{{\n  \"device\": {:?},\n  \"mode\": {:?},\n  \"full_oracle\": {:?},\n  \"capabilities\": {{\"mixed_supported\": {}, \"full_fp16_supported\": {}, \"subgroup_size\": {}, \"driver_default_subgroup_size\": {}}},\n  \"dispatch_counts\": {{\"cooperative\": {}, \"plain_f16\": {}, \"shape_fallbacks\": {}}},\n  \"gates\": {}, \"failed\": {},\n  \"results\": [\n{}\n  ]\n}}",
            device.name(),
            mode.label(),
            if full_accumulation == OracleAccumulation::Fp16Dot2 {"rdna3-dot2"} else {"tile16"},
            capabilities.mixed_supported(),
            capabilities.full_fp16_supported(),
            hierarchos_vulkan::wmma::WMMA_SUBGROUP_SIZE,
            capabilities.subgroup_size,
            counts.cooperative,
            counts.plain_f16,
            counts.shape_fallbacks,
            all.len(),
            failed,
            rows.join(",\n")
        );
    } else {
        println!("WMMA numerical qualification");
        println!("device: {}", device.name());
        println!("mode: {} ({})", mode.label(), mode.description());
        print!("{}", capabilities.diagnostics());
        println!("dispatch counts: {counts:?}");
        println!("matched-oracle and FP32 drift:");
        for gate in &all {
            println!("{}", gate.render());
        }
        println!("gates: {} total, {} failed", all.len(), failed);
        if !unqualified.is_empty() {
            println!("unqualified gates:");
            for label in &unqualified {
                println!("  {label}");
            }
        }
    }

    if failed > 0 {
        bail!("{failed} of {} numerical gates failed", all.len());
    }
    println!("status: PASS");
    Ok(())
}
