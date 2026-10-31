//! Standalone shader parity/benchmark harness. Compile with the crate's cached
//! anyhow/ash/windows-sys dependencies and OUT_DIR shader registry, or include
//! as a Cargo example. Arguments: [BASELINE_SHADER_DIR] [--bench] [--golden FILE]
//! [--write-golden FILE].
//!
//! The baseline directory is optional when a golden file is used: recording each
//! invocation's output digest lets CI catch an attention regression with no
//! pre-change artifacts on the machine.
#![allow(dead_code)]
#[path = "../src/kernel_golden.rs"]
mod kernel_golden;
#[path = "../src/vulkan.rs"]
mod vulkan;
use anyhow::{bail, Result};
use kernel_golden::{coverage_args, Golden};
use std::path::Path;
use vulkan::{ComputeBatch, ComputeKernel, GpuBuffer, VulkanDevice};

#[derive(Clone)]
struct Case {
    name: String,
    // The two words past the shared attention block are the key-range block a
    // backward submission owns: its index, then the count. A count of one is the
    // unsplit dispatch, and the forward kernel ignores both.
    p: [u32; 32],
}

impl Case {
    fn base(name: &str, query: u32, key: u32, qdim: u32, vdim: u32) -> Self {
        let mut p = [0; 32];
        p[0] = 2;
        p[1] = query;
        p[2] = key;
        p[3] = 4 * qdim;
        p[4] = 2 * vdim;
        p[5] = 4;
        p[6] = 2;
        p[8] = 0.7f32.to_bits();
        p[9] = 0.8f32.to_bits();
        p[12] = 8f32.to_bits();
        p[14] = 3;
        p[15] = 7;
        p[16] = 19;
        p[18] = 1;
        p[20] = 16;
        p[21] = 128;
        p[27] = 1;
        p[28] = key;
        Self {
            name: name.into(),
            p,
        }
    }
}

fn values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 73 + salt * 31) % 251) as f32 - 125.0) / 127.0)
        .collect()
}

/// Key-range blocks this invocation drives. Only the strict adjoint may be
/// split: its per-key K/V outputs are self-contained assignments, so blocking
/// keys preserves every reduction order. The fused path accumulates a running
/// softmax_dot across keys and stays single-block, matching the shader's own
/// gate, so a caller that asks for a split of it still gets one block.
fn key_blocks_for(case: &Case, backward: bool, requested: u32) -> u32 {
    if !backward || requested <= 1 {
        return 1;
    }
    match case.p[29] {
        1 | 2 | 9 => requested,
        _ => 1,
    }
}

fn run(
    device: &VulkanDevice,
    dir: &Path,
    case: &Case,
    backward: bool,
    repeats: usize,
    split_blocks: u32,
) -> Result<(Vec<f32>, f64)> {
    let p = &case.p;
    let (batch, query, key, qhidden, vhidden, heads, kvheads) = (
        p[0] as usize,
        p[1] as usize,
        p[2] as usize,
        p[3] as usize,
        p[4] as usize,
        p[5] as usize,
        p[6] as usize,
    );
    let keyhidden = qhidden / heads * kvheads;
    let outhidden = vhidden / kvheads * heads;
    let q = GpuBuffer::from_f32(device, &values(batch * query * qhidden, 1))?;
    let mut keys = values(batch * key * keyhidden, 2);
    let mut vals = values(batch * key * vhidden, 3);
    let pages: Vec<u32> = (0..batch * key)
        .map(|i| (batch * key - 1 - i) as u32)
        .collect();
    if p[26] != 0 {
        let original_k = keys.clone();
        let original_v = vals.clone();
        for (logical, &physical) in pages.iter().enumerate() {
            keys[physical as usize * keyhidden..(physical as usize + 1) * keyhidden]
                .copy_from_slice(&original_k[logical * keyhidden..(logical + 1) * keyhidden]);
            vals[physical as usize * vhidden..(physical as usize + 1) * vhidden]
                .copy_from_slice(&original_v[logical * vhidden..(logical + 1) * vhidden]);
        }
    }
    let k = GpuBuffer::from_f32(device, &keys)?;
    let v = GpuBuffer::from_f32(device, &vals)?;
    let qm = GpuBuffer::from_f32(
        device,
        &(0..batch * query)
            .map(|i| if i % 7 == 2 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>(),
    )?;
    let local_key = key - p[24] as usize;
    let km = GpuBuffer::from_f32(
        device,
        &(0..batch * local_key)
            .map(|i| if i % 5 == 0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>(),
    )?;
    let alibi = GpuBuffer::from_f32(
        device,
        &(0..batch * key)
            .map(|i| (i % key) as f32 - key as f32)
            .collect::<Vec<_>>(),
    )?;
    let sink = GpuBuffer::from_f32(device, &values(heads, 4))?;
    let relative = GpuBuffer::from_f32(device, &values(p[20] as usize * heads, 5))?;
    let sparse = GpuBuffer::from_f32(
        device,
        &(0..batch * kvheads * query * key)
            .map(|i| if i % 4 == 1 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>(),
    )?;
    let page = GpuBuffer::from_u32(device, &pages)?;
    let out = GpuBuffer::zeros_f32(device, batch * query * outhidden)?;
    let go = GpuBuffer::from_f32(device, &values(batch * query * outhidden, 6))?;
    let gq = GpuBuffer::zeros_f32(device, batch * query * qhidden)?;
    let gk = GpuBuffer::zeros_f32(device, batch * key * keyhidden)?;
    let gv = GpuBuffer::zeros_f32(device, batch * key * vhidden)?;
    let gs = GpuBuffer::zeros_f32(device, heads)?;
    let gr = GpuBuffer::zeros_f32(device, p[20] as usize * heads)?;
    let stem = if backward {
        "transformer_attention_backward.spv"
    } else {
        "transformer_attention_forward.spv"
    };
    let kernel = ComputeKernel::new(
        device,
        &std::fs::read(dir.join(stem))?,
        if backward { 15 } else { 11 },
        if backward { 128 } else { 120 },
    )?;
    let bindings: Vec<&GpuBuffer> = if backward {
        vec![
            &q, &k, &v, &qm, &km, &alibi, &go, &gq, &gk, &gv, &sink, &gs, &relative, &gr, &sparse,
        ]
    } else {
        vec![
            &q, &k, &v, &qm, &km, &alibi, &out, &sink, &relative, &sparse, &page,
        ]
    };
    // One key-range block per submission. The watchdog bounds a submission, so a
    // block only shortens anything when it is submitted on its own, and it carries
    // its identity in the two words the push block gained for it. Every submission
    // therefore launches a z of one: the grid no longer partitions anything.
    let blocks = key_blocks_for(case, backward, split_blocks);
    let record = |commands: &mut ComputeBatch, block: u32| -> Result<()> {
        let mut words = *p;
        if backward {
            words[30] = block;
            words[31] = blocks;
        }
        let bytes = if backward { 128 } else { 120 };
        let push = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, bytes) };
        let dispatch = if backward {
            [kvheads as u32, batch as u32, 1]
        } else {
            [query as u32, heads as u32, batch as u32]
        };
        kernel.record_dispatch(commands, &bindings, push, dispatch)
    };
    // Warm the pipeline before measuring command submission plus GPU execution.
    let mut warm = ComputeBatch::new(device)?;
    record(&mut warm, 0)?;
    warm.submit()?;
    // Median GPU timestamp delta, so the number is independent of host-side
    // command-recording cost in an unoptimized build. Falls back to wall time
    // when the driver does not report a single profile sample.
    let mut ms = 0.0;
    for block in 0..blocks {
        let mut samples = Vec::new();
        for _ in 0..5 {
            let total_ms = device.time_compute_batch_ms(|commands| {
                for _ in 0..repeats {
                    record(commands, block)?;
                }
                Ok(())
            })?;
            samples.push(total_ms / repeats as f64);
        }
        samples.sort_by(f64::total_cmp);
        let block_ms = samples[samples.len() / 2];
        if blocks > 1 {
            println!("split_block {}/{blocks} ms={block_ms:.3}", block + 1);
        }
        ms += block_ms;
    }
    // A split run has to drive every block once more, because each block owns its own
    // key range and the reads below need all of them. An unsplit run already holds the
    // complete result from the timed submissions, so it skips this: the relative-bias
    // gradient accumulates across dispatches, which makes the recorded digests depend
    // on the dispatch count, and an extra pass would move those digests for a reason
    // that has nothing to do with the module under test.
    if blocks > 1 {
        let mut final_commands = ComputeBatch::new(device)?;
        for block in 0..blocks {
            record(&mut final_commands, block)?;
            if block + 1 < blocks {
                final_commands.submit_and_restart()?;
            }
        }
        final_commands.submit()?;
    }
    let mut result = if backward {
        let mut r = gq.read_f32(batch * query * qhidden)?;
        r.extend(gk.read_f32(batch * key * keyhidden)?);
        r.extend(gv.read_f32(batch * key * vhidden)?);
        if p[17] != 0 {
            r.extend(gs.read_f32(heads)?);
        }
        if p[19] != 0 {
            r.extend(gr.read_f32(p[20] as usize * heads)?);
        }
        r
    } else {
        out.read_f32(batch * query * outhidden)?
    };
    // Relative-bias gradients intentionally accumulate across dispatches; each
    // parity run uses identical counts and separately allocated zero buffers.
    result.shrink_to_fit();
    Ok((result, ms))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: attention_kernel_cache_check [BASELINE_SHADER_DIR] [--bench [--bench-only] [--bench-mode 0|1]] [--candidate-dir DIR] [--bench-repeats N] [--split-key-blocks N] [--golden FILE | --write-golden FILE]");
    }
    let device = VulkanDevice::new()?;
    // A leading non-flag argument is the baseline artifact directory. Golden
    // mode works without it, because recorded digests replace the comparison.
    let baseline = args
        .get(1)
        .filter(|arg| !arg.starts_with("--"))
        .map(|arg| Path::new(arg.as_str()));
    let candidate = args
        .iter()
        .position(|arg| arg == "--candidate-dir")
        .map(|index| Path::new(&args[index + 1]))
        .unwrap_or(Path::new("shaders"));
    // The backward benchmark submits a long sequence of near-watchdog-sized
    // scalar dispatches; allowing an override keeps the harness usable on
    // drivers that reset the device, and pairs with --candidate-dir for
    // unoptimized-vs-unoptimized control runs.
    let bench_repeats = args
        .iter()
        .position(|arg| arg == "--bench-repeats")
        .map(|index| args[index + 1].parse::<usize>().expect("bench repeats"));
    if bench_repeats == Some(0) {
        bail!("--bench-repeats must be positive");
    }
    // Sequential key-range dispatches: the backward grid's third dimension
    // partitions the key range so one invocation covers one key block.
    let split_blocks = args
        .iter()
        .position(|arg| arg == "--split-key-blocks")
        .map(|index| args[index + 1].parse::<u32>().expect("split key blocks"))
        .unwrap_or(1);
    if split_blocks == 0 {
        bail!("--split-key-blocks must be positive");
    }
    let (golden_compare, golden_write, golden_strict) = kernel_golden::golden_flags(&args)?;
    // Coverage is audited or recorded whenever a golden is involved; `--bench-only` cannot
    // restrict the golden set (it is rejected when a golden is present), so an audit is
    // always the whole harness.
    let coverage = coverage_args(&args)?;
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
    if baseline.is_none() && golden.is_none() {
        bail!("attention_kernel_cache_check needs a baseline shader directory or a golden file");
    }
    let bench_only = args.iter().any(|arg| arg == "--bench-only");
    if bench_only && (golden.is_some() || !args.iter().any(|arg| arg == "--bench")) {
        bail!("--bench-only requires --bench and cannot skip golden verification");
    }
    let mut cases = Vec::new();
    for mode in [0, 1, 2, 3, 9] {
        for variant in 0..7 {
            let mut c = Case::base(&format!("mode{mode}_feature{variant}"), 9, 11, 5, 3);
            c.p[29] = mode;
            match variant {
                1 => {
                    c.p[7] = 3;
                    c.p[13] = 0.2f32.to_bits();
                    c.p[10] = 1.1f32.to_bits();
                }
                2 => {
                    c.p[18] = 0;
                    c.p[7] = 2;
                    c.p[11] = 1;
                    c.p[17] = 1;
                }
                3 => {
                    c.p[11] = 2;
                    c.p[19] = 1;
                    c.p[23] = 2;
                }
                4 => {
                    c.p[18] = 2;
                    c.p[11] = 3;
                    c.p[13] = 0.2f32.to_bits();
                }
                5 => {
                    c.p[24] = 3;
                    c.p[25] = 2;
                    c.p[7] = 3;
                    c.p[23] = 1;
                }
                6 => {
                    c.p[26] = 1;
                    c.p[22] = 2;
                    c.p[17] = 1;
                }
                _ => {}
            }
            cases.push(c);
        }
    }
    for (name, query, key, dim, mode) in [
        ("key_cache_512", 3, 512, 2, 0),
        ("key_cache_513", 3, 513, 2, 0),
        ("forward_cache_1024", 2, 1024, 2, 0),
        ("forward_cache_1025", 2, 1025, 2, 0),
        ("strict_head_fallback", 3, 4, 257, 1),
        ("strict_stats_fallback", 129, 4, 2, 1),
    ] {
        let mut c = Case::base(name, query, key, dim, dim);
        c.p[18] = 0;
        c.p[29] = mode;
        cases.push(c);
    }
    let mut window = Case::base("long_input_short_window", 2, 1050, 3, 2);
    window.p[22] = 1024;
    window.p[7] = 3;
    cases.push(window);
    for c in &cases {
        if bench_only {
            break;
        }
        for backward in [false, true] {
            if backward && c.p[26] != 0 {
                continue;
            }
            if key_blocks_for(c, backward, split_blocks) > 1 {
                println!(
                    "split {} backward={} blocks={}",
                    c.name, backward, split_blocks
                );
            }
            let (new, _) = run(&device, candidate, c, backward, 1, split_blocks)?;
            if let Some(baseline) = baseline {
                let (old, _) = run(&device, baseline, c, backward, 1, split_blocks)?;
                if old.len() != new.len() {
                    bail!("{}: output lengths differ", c.name);
                }
                let mut different = 0usize;
                let mut max_abs = 0f32;
                for (i, (&a, &b)) in old.iter().zip(&new).enumerate() {
                    if !a.is_finite() || !b.is_finite() {
                        bail!("{} backward={backward}: nonfinite output at {i}", c.name);
                    }
                    if a.to_bits() != b.to_bits() {
                        different += 1;
                    }
                    max_abs = max_abs.max((a - b).abs());
                    if (a - b).abs() > 2e-6 + 2e-6 * a.abs() {
                        bail!(
                            "{} backward={backward}: mismatch at {i}: old={a} new={b}",
                            c.name
                        );
                    }
                }
                println!(
                    "parity {} backward={} different={} max_abs={:.9}",
                    c.name, backward, different, max_abs
                );
                if different != 0 {
                    bail!(
                        "{} backward={backward}: {different} outputs are not bit-identical",
                        c.name
                    );
                }
            } else {
                println!("parity {} backward={} digest-only", c.name, backward);
            }
            if let Some(golden) = golden.as_mut() {
                golden.observe(&format!("{} backward={backward}", c.name), &new);
            }
        }
    }
    if let Some(golden) = golden {
        golden.finish()?;
    }
    let Some(baseline) = baseline else {
        return Ok(());
    };
    if args.iter().any(|a| a == "--bench") {
        // The backward kernel launches one workgroup per (KV head, batch row),
        // so its shape decides how much of the device can be busy at once.
        // These overrides let one run separate "not enough workgroups" from
        // "too much serial work per workgroup".
        let flag = |name: &str, default: u32| {
            args.iter()
                .position(|arg| arg == name)
                .map(|index| args[index + 1].parse::<u32>().expect(name))
                .unwrap_or(default)
        };
        let bench_query = flag("--bench-query", 64);
        let bench_key = flag("--bench-key", 64);
        let bench_batch = flag("--bench-batch", 2);
        let bench_heads = flag("--bench-heads", 4);
        let bench_kv_heads = flag("--bench-kv-heads", 2);
        // Head dimension is the knob that decides which strict path the shader
        // takes: at or below the 256-wide cache bound the strict adjoint reuses
        // its row statistics, above it the fallback replays them per head
        // dimension element. Overriding it measures that cliff directly.
        let bench_qk_dim = flag("--bench-qk-dim", 32);
        let bench_value_dim = flag("--bench-value-dim", 32);
        let bench_mode = args
            .iter()
            .position(|arg| arg == "--bench-mode")
            .map(|index| args[index + 1].parse::<u32>().expect("bench mode"));
        if bench_mode.is_some_and(|mode| mode > 1) {
            bail!("--bench-mode must be 0 or 1");
        }
        for mode in [0, 1] {
            if bench_mode.is_some_and(|selected| selected != mode) {
                continue;
            }
            let mut c = Case::base(
                &format!("bench_mode{mode}"),
                bench_query,
                bench_key,
                bench_qk_dim,
                bench_value_dim,
            );
            c.p[0] = bench_batch;
            c.p[5] = bench_heads;
            c.p[6] = bench_kv_heads;
            c.p[29] = mode;
            for backward in [false, true] {
                let workgroups = if backward {
                    bench_kv_heads * bench_batch
                } else {
                    bench_query * bench_heads * bench_batch
                };
                println!(
                    "bench_shape {} backward={backward} workgroups={workgroups} query={bench_query} key={bench_key} batch={bench_batch} heads={bench_heads} kv_heads={bench_kv_heads}",
                    c.name
                );
                let repeats = bench_repeats.unwrap_or(if backward && mode != 0 { 1 } else { 5 });
                println!("timing {} backward={backward} repeats={repeats}", c.name);
                let (old_values, old) =
                    run(&device, baseline, &c, backward, repeats, split_blocks)?;
                let (new_values, new) =
                    run(&device, candidate, &c, backward, repeats, split_blocks)?;
                if old_values.len() != new_values.len()
                    || old_values.iter().zip(&new_values).any(|(a, b)| {
                        !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits()
                    })
                {
                    bail!(
                        "{} backward={backward}: benchmark outputs are not bit-identical",
                        c.name
                    );
                }
                println!(
                    "benchmark {} backward={} baseline_ms={:.3} optimized_ms={:.3} speedup={:.2}",
                    c.name,
                    backward,
                    old,
                    new,
                    old / new
                );
            }
        }
    }
    Ok(())
}
