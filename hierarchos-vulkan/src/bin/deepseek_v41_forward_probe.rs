//! Times the dedicated DeepSeek V4.1 forward graph so its kernels can be ranked
//! by GPU time before any optimization is attempted.
//!
//! This is the harness the Intel Gen9 handoff brief asks for: profile first, then
//! pick the hot projection and attention stages. Run it with
//! `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` and feed stderr to
//! `validation/rank_kernel_profile.py`:
//!
//! ```text
//! $env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"
//! cargo run --release --bin hierarchos-vulkan-deepseek-v41-forward-probe -- --steps 3 2> profile.log
//! python validation/rank_kernel_profile.py profile.log --min-dispatches 512
//! ```
//!
//! The probe prints one machine-readable `v41_probe_*` line per phase plus a
//! digest of the last-position logits, so a kernel change can be shown to leave
//! the forward bit-identical while the profile shows what it bought.
//!
//! `--residency resident` keeps every weight resident instead of streaming it
//! through the checkpoint, which isolates compute from checkpoint I/O; the
//! default mirrors the shipping loader exactly.

use anyhow::{anyhow, bail, Context, Result};
use std::path::PathBuf;
use std::time::Instant;

use hierarchos_vulkan::deepseek_v41::{
    DeepseekV41AttentionRuntimeConfig, DeepseekV41CausalLM, DeepseekV41Checkpoint,
    DeepseekV41EngramResidency, DeepseekV41EngramTokens, DeepseekV41ExpertResidency,
    DeepseekV41ProjectionResidency, DeepseekV41VocabularyResidency, DeepseekV41WeightOutput,
};
use hierarchos_vulkan::VulkanDevice;

struct Args {
    model: PathBuf,
    prefill: usize,
    decode: usize,
    steps: usize,
    warmup: usize,
    residency: Residency,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Residency {
    /// Exactly the shipping `from_hf_package` loader.
    Default,
    /// Every table resident, so the profile is compute rather than checkpoint I/O.
    Resident,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("validation/fixtures/deepseek_v41_joint_4141111"),
        prefill: 8,
        decode: 8,
        steps: 3,
        warmup: 1,
        residency: Residency::Default,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = |name: &str| -> Result<String> {
            argv.next()
                .ok_or_else(|| anyhow!("{name} requires a value"))
        };
        match flag.as_str() {
            "--model" => args.model = PathBuf::from(value("--model")?),
            "--prefill" => args.prefill = value("--prefill")?.parse()?,
            "--decode" => args.decode = value("--decode")?.parse()?,
            "--steps" => args.steps = value("--steps")?.parse()?,
            "--warmup" => args.warmup = value("--warmup")?.parse()?,
            "--residency" => {
                args.residency = match value("--residency")?.as_str() {
                    "default" => Residency::Default,
                    "resident" => Residency::Resident,
                    other => bail!("--residency accepts default|resident, got {other}"),
                }
            }
            "--help" | "-h" => {
                println!(
                    "usage: deepseek_v41_forward_probe [--model DIR] [--prefill N] [--decode N] \
                     [--steps N] [--warmup N] [--residency default|resident]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}"),
        }
    }
    if args.prefill == 0 || args.decode == 0 || args.steps == 0 {
        bail!("--prefill, --decode and --steps must be positive");
    }
    Ok(args)
}

/// FNV-1a over the order-sensitive bit pattern of a logit row.
fn digest(values: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timed samples are finite"));
    let mid = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

fn load(device: VulkanDevice, args: &Args) -> Result<DeepseekV41CausalLM> {
    match args.residency {
        Residency::Default => DeepseekV41CausalLM::from_hf_package(
            device,
            &args.model,
            1,
            args.prefill + args.decode + 8,
        ),
        Residency::Resident => {
            let mut checkpoint = DeepseekV41Checkpoint::open(&args.model)?;
            let tokens: DeepseekV41EngramTokens = serde_json::from_slice(
                &std::fs::read(args.model.join("engram_tokens.json"))
                    .context("reading the resident probe's engram_tokens.json")?,
            )?;
            let precision = match checkpoint.tensor_dtype("embed.weight")?.as_str() {
                "BF16" => DeepseekV41WeightOutput::Bf16,
                "F32" => DeepseekV41WeightOutput::Fp32,
                other => bail!("V4.1 embedding requires BF16/F32 storage, got {other}"),
            };
            DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
                device,
                &mut checkpoint,
                DeepseekV41AttentionRuntimeConfig {
                    batch_size: 1,
                    max_prefill: args.prefill,
                    max_positions: args.prefill + args.decode + 8,
                },
                Some(tokens),
                precision,
                DeepseekV41ExpertResidency::Resident,
                DeepseekV41EngramResidency::Resident,
                DeepseekV41ProjectionResidency::Resident,
                DeepseekV41VocabularyResidency::Resident,
            )
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let tokens: Vec<u32> = (0..args.prefill as u32).collect();

    let started = Instant::now();
    let device = VulkanDevice::new()?;
    let mut model = load(device.clone(), &args)?;
    eprintln!(
        "v41_probe_load ms={:.3} submissions={} device={:?} vendor=0x{:04x}",
        started.elapsed().as_secs_f64() * 1e3,
        device.submission_count(),
        device.name(),
        device.vendor_id(),
    );

    let mut prefill_ms = Vec::new();
    let mut decode_ms = Vec::new();
    let mut latest = None;
    for step in 0..args.warmup + args.steps {
        if step == args.warmup {
            eprintln!(
                "v41_probe_io phase=warmup_complete json={}",
                serde_json::to_string(&model.checkpoint_io_stats()?)?
            );
        }
        let measured = step >= args.warmup;
        let start = Instant::now();
        let submissions = device.submission_count();
        let prefill = model.prefill(&tokens, tokens.len(), None)?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if measured {
            prefill_ms.push(elapsed);
        }
        eprintln!(
            "v41_probe_phase phase=prefill step={step} measured={measured} ms={elapsed:.3} \
             rows={} vocab={} submissions={}",
            prefill.seq_len,
            prefill.vocab_size,
            device.submission_count() - submissions,
        );

        let mut logits = prefill;
        for offset in 0..args.decode {
            let position = args.prefill + offset;
            let token = [tokens[offset % tokens.len()]];
            let start = Instant::now();
            let submissions = device.submission_count();
            logits = model.decode(&token, position, None)?;
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            if measured {
                decode_ms.push(elapsed);
            }
            eprintln!(
                "v41_probe_phase phase=decode step={step} position={position} \
                 measured={measured} ms={elapsed:.3} submissions={}",
                device.submission_count() - submissions,
            );
        }
        let row = logits.read_all()?;
        if row.iter().any(|value| !value.is_finite()) {
            bail!("V4.1 probe produced a non-finite logit at step {step}");
        }
        eprintln!(
            "v41_probe_logits step={step} count={} digest={:016x}",
            row.len(),
            digest(&row),
        );
        latest = Some((row, logits.vocab_size));
    }

    eprintln!(
        "v41_probe_io phase=complete json={}",
        serde_json::to_string(&model.checkpoint_io_stats()?)?
    );
    let (row, vocab) = latest.expect("at least one measured step");
    let decode_median = median(decode_ms);
    println!(
        "v41_probe_summary prefill_tokens={} decode_tokens={} steps={} residency={} \
         prefill_ms_median={:.3} decode_ms_median={:.3} decode_ms_per_token={:.3} \
         vocab={} logits={} digest={:016x}",
        args.prefill,
        args.decode,
        args.steps,
        if args.residency == Residency::Default {
            "default"
        } else {
            "resident"
        },
        median(prefill_ms),
        decode_median,
        // Each sample times one decode() call with one token. The --decode
        // argument controls sample count, not the number of tokens in a sample.
        decode_median,
        vocab,
        row.len(),
        digest(&row),
    );
    Ok(())
}
