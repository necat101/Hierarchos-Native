"""Sequential stable/candidate Transformer inference, startup and allocation A/B.

Builds the SAME observation-only Rust program against each library. All timed
calls use the public GPU inference API, including its ordinary readback. Output
bits must remain identical across every invocation within a leg. The separate
stable-logit gate owns cross-version numerical qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from check_stable_logits import identity
from check_stable_probe_parity import write_source

CRATE = Path(__file__).resolve().parents[1]
BINARY = "transformer-inference-observer"
SOURCE = r'''
use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{VulkanDevice, VulkanTransformer, VulkanSeq2SeqTransformer};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Fixture {
    batch_size: usize, seq_len: usize, input_ids: Vec<u32>, attention_mask: Vec<f32>,
    token_type_ids: Option<Vec<u32>>, encoder_input_ids: Option<Vec<u32>>,
    encoder_attention_mask: Option<Vec<f32>>,
}
enum Graph { Decoder(Box<VulkanTransformer>), Seq2Seq(Box<VulkanSeq2SeqTransformer>) }
impl Graph {
    fn forward(&self, f: &Fixture) -> Result<Vec<f32>> {
        match self {
            Self::Decoder(g) => g.forward_logits_with_token_type_ids(
                &f.input_ids, f.token_type_ids.as_deref(), &f.attention_mask),
            Self::Seq2Seq(g) => g.forward_logits(f.encoder_input_ids.as_deref().unwrap(),
                f.encoder_attention_mask.as_deref(), &f.input_ids, &f.attention_mask),
        }
    }
}
fn main() -> Result<()> {
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(run)?.join()
        .map_err(|_| anyhow::anyhow!("Transformer benchmark worker panicked"))?
}
fn run() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 4 { bail!("model fixture warmup iterations are required"); }
    let model = PathBuf::from(&args[0]);
    let fixture: Fixture = serde_json::from_slice(&fs::read(PathBuf::from(&args[1]))?)?;
    let warmup: usize = args[2].to_string_lossy().parse()?;
    let iterations: usize = args[3].to_string_lossy().parse()?;
    if warmup == 0 || iterations < 2 { bail!("warmup >= 1 and iterations >= 2 required"); }
    let started = Instant::now();
    let device = VulkanDevice::new()?;
    let device_setup_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let graph = if let Some(ids) = fixture.encoder_input_ids.as_ref() {
        Graph::Seq2Seq(Box::new(VulkanSeq2SeqTransformer::from_hf_package_with_batch_size(
            device.clone(), &model, fixture.batch_size, ids.len() / fixture.batch_size, fixture.seq_len)?))
    } else {
        Graph::Decoder(Box::new(VulkanTransformer::from_hf_package(
            device.clone(), &model, fixture.batch_size, fixture.seq_len)?))
    };
    let graph_setup_ms = started.elapsed().as_secs_f64() * 1000.0;
    let before = device.memory_stats()?;
    let expected = graph.forward(&fixture)?;
    if expected.is_empty() || expected.iter().any(|v| !v.is_finite()) { bail!("empty/non-finite logits"); }
    let verify = |actual: &[f32]| -> Result<()> {
        if actual.len() != expected.len() || actual.iter().zip(&expected).any(|(a,b)| a.to_bits()!=b.to_bits()) {
            bail!("repeated forward changed a logit bit");
        }
        Ok(())
    };
    for _ in 1..warmup { verify(&graph.forward(&fixture)?)?; }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let actual = graph.forward(&fixture)?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        verify(&actual)?;
    }
    let after = device.memory_stats()?;
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let mut digest = Sha256::new();
    for value in &expected { digest.update(value.to_bits().to_le_bytes()); }
    println!("{}", serde_json::json!({
        "device": device.name(), "vendor_id": device.vendor_id(), "device_id": device.device_id(),
        "driver_version": device.driver_version(), "batch": fixture.batch_size, "tokens": fixture.seq_len,
        "device_setup_ms": device_setup_ms, "graph_setup_ms": graph_setup_ms,
        "warmup": warmup, "iterations": iterations, "samples_ms": samples,
        "median_ms": sorted[sorted.len()/2], "min_ms": sorted[0], "max_ms": sorted[sorted.len()-1],
        "logit_values": expected.len(), "logits_fp32_sha256": format!("{:x}",digest.finalize()),
        "memory_before": {"buffers":before.live_buffer_count,"bytes":before.live_buffer_bytes,
            "driver_allocations":before.driver_allocation_count,"reserved_bytes":before.reserved_bytes},
        "memory_after": {"buffers":after.live_buffer_count,"bytes":after.live_buffer_bytes,
            "driver_allocations":after.driver_allocation_count,"reserved_bytes":after.reserved_bytes}
    }));
    Ok(())
}
'''


def build_observer(crate: Path, label: str) -> Path:
    observer = CRATE / "target" / "transformer-inference-observers" / label
    observer.mkdir(parents=True, exist_ok=True)
    write_source(observer / "main.rs", SOURCE)
    write_source(observer / "Cargo.toml", '[package]\nname = "transformer-inference-observer"\n'
        'version = "0.0.0"\nedition = "2021"\n[workspace]\n[[bin]]\n'
        f'name = "{BINARY}"\npath = "main.rs"\n[dependencies]\nanyhow = "1"\n'
        'serde = {version="1", features=["derive"]}\nserde_json = "1"\nsha2 = "0.10"\n'
        f'hierarchos-vulkan = {{path={json.dumps(crate.as_posix())}}}\n'
        '[profile.release]\nopt-level = 3\nlto = "thin"\ncodegen-units = 1\n')
    lock = observer / "Cargo.lock"
    if not lock.exists():
        lock.write_bytes((crate / "Cargo.lock").read_bytes())
    command = ["cargo", "build", "--manifest-path", str(observer / "Cargo.toml"), "--release",
               "--target-dir", str(crate / "target"), "--message-format=json-render-diagnostics", "-j", "1"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-5000:])
    for line in completed.stdout.splitlines():
        item = json.loads(line)
        if item.get("reason") == "compiler-artifact" and item.get("executable"):
            return Path(item["executable"])
    raise RuntimeError("Cargo did not return a Transformer observer executable")


def power_scheme() -> str:
    return subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip() if os.name == "nt" else "not Windows"


def cases(fixtures: Path, output: Path):
    rows = []
    for family, case in (("gpt2", "unmasked"), ("gpt2_wide", "unmasked"),
                         ("gpt2_wide", "padded"), ("llama_wide", "unmasked"), ("llama_wide", "padded")):
        package = fixtures / family / case
        rows.append((f"{family}-{case}", package, package / "vulkan_logits_fixture.json"))
    package = fixtures / "gpt2_wide" / "unmasked"
    base = json.loads((package / "vulkan_logits_fixture.json").read_text())
    for batch, tokens, name in ((1, 1, "gpt2-wide-decode"), (32, base["seq_len"], "gpt2-wide-batch32")):
        fixture = dict(base, batch_size=batch, seq_len=tokens)
        for key in ("input_ids", "attention_mask"):
            fixture[key] = (base[key] * (batch * tokens // len(base[key]) + 1))[:batch * tokens]
        path = output / f"{name}.json"
        path.write_text(json.dumps(fixture) + "\n")
        rows.append((name, package, path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    for key in ("HIERARCHOS_VULKAN_FORCE_VENDOR", "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS"):
        if key in os.environ:
            parser.error(f"unset {key}; this benchmark measures the shipping default")
    if args.warmup < 1 or args.iterations < 2 or args.repeats < 3:
        parser.error("require warmup >= 1, iterations >= 2 and at least three alternating pairs")
    stable = args.stable.resolve()
    identities = {"stable": identity(stable), "candidate": identity(CRATE.parent)}
    if identities["stable"]["status"]:
        parser.error("stable checkout must be clean")
    binaries = {"stable": build_observer(stable / "hierarchos-vulkan", "stable"),
                "candidate": build_observer(CRATE, "candidate")}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    power = power_scheme()
    rows = []
    for name, model, fixture in cases(args.fixture_root.resolve(), args.output):
        runs = []
        for repeat in range(args.repeats):
            pair = {}
            for label in (("stable", "candidate") if repeat % 2 == 0 else ("candidate", "stable")):
                completed = subprocess.run([str(binaries[label]), str(model), str(fixture),
                    str(args.warmup), str(args.iterations)], capture_output=True, text=True, check=True)
                report = json.loads(completed.stdout)
                if report["vendor_id"] != 0x1002:
                    raise ValueError("Transformer benchmark did not execute on AMD")
                (args.output / f"{name}-{repeat}-{label}.json").write_text(json.dumps(report, indent=2) + "\n")
                pair[label] = report
            pair["candidate_over_stable"] = pair["candidate"]["median_ms"] / pair["stable"]["median_ms"]
            runs.append(pair)
            print(name, repeat, pair["candidate_over_stable"], flush=True)
        rows.append({"case": name, "model_weights_sha256": hashlib.sha256((model / "model.safetensors").read_bytes()).hexdigest(),
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "median_candidate_over_stable": statistics.median(run["candidate_over_stable"] for run in runs), "runs": runs})
        (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
    if identities != {"stable": identity(stable), "candidate": identity(CRATE.parent)}:
        raise ValueError("sources changed during Transformer A/B")
    if hashes != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}:
        raise ValueError("executable changed during Transformer A/B")
    after = power_scheme()
    if power != after:
        raise ValueError("power mode changed during Transformer A/B")
    (args.output / "receipt.json").write_text(json.dumps({"identities": identities, "binary_sha256": hashes,
        "power_before": power, "power_after": after, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
