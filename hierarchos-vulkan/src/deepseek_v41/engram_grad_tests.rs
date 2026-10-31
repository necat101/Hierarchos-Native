use super::*;
use crate::deepseek_v41::peft::DeepseekV41LoraHost;
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    tolerance: f64,
    rtol: f64,
    comparison: String,
    gates: Vec<GateCase>,
    chains: Vec<Chain>,
}
#[derive(Deserialize)]
struct GateCase {
    name: String,
    rows: usize,
    hc: usize,
    hidden: usize,
    eps: f32,
    precision: String,
    mask: Option<Vec<u32>>,
    stream_bits: Vec<u32>,
    kv_bits: Vec<u32>,
    weight_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    output_bits: Vec<u32>,
    grad_stream_bits: Vec<u32>,
    grad_kv_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Table {
    dtype: String,
    values_bits: Vec<u32>,
    data: Vec<u8>,
    scales: Vec<u8>,
}
#[derive(Deserialize)]
struct Projection {
    rows: usize,
    cols: usize,
    dtype: String,
    #[serde(default)]
    values_bits: Vec<u32>,
    data: Vec<u8>,
    scales: Vec<u8>,
    rank: usize,
    scale: f32,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
impl Projection {
    fn native(&self, adapter: bool) -> Result<DeepseekV41LinearWeights> {
        let lora = (adapter && self.rank != 0).then(|| DeepseekV41LoraHost {
            rank: self.rank,
            scale: self.scale,
            a: values(&self.a_bits).into(),
            b: values(&self.b_bits).into(),
        });
        Ok(match self.dtype.as_str() {
            "fp32" => DeepseekV41LinearWeights::Fp32 {
                rows: self.rows,
                cols: self.cols,
                values: values(&self.values_bits),
                lora,
            },
            "bf16" => DeepseekV41LinearWeights::Bf16 {
                rows: self.rows,
                cols: self.cols,
                values: values(&self.values_bits),
                lora,
            },
            "fp8_block32" => DeepseekV41LinearWeights::Packed {
                weight: DeepseekV41PackedWeights::new(
                    self.rows,
                    self.cols,
                    DeepseekV41WeightFormat::Fp8Block32,
                    self.data.clone(),
                    self.scales.clone(),
                )?,
                lora,
            },
            _ => bail!("unsupported Engram fixture projection"),
        })
    }
}
#[derive(Deserialize)]
struct Chain {
    name: String,
    rows: usize,
    hc: usize,
    hidden: usize,
    cols: usize,
    head_dim: usize,
    table_rows: usize,
    eps: f32,
    precision: String,
    table: Table,
    projection: Projection,
    hashes: Vec<u32>,
    mask: Vec<u32>,
    stream_bits: Vec<u32>,
    weight_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    output_bits: Vec<u32>,
    grad_stream_bits: Vec<u32>,
    grad_kv_bits: Vec<u32>,
    grad_embedded_bits: Vec<u32>,
    grad_a_bits: Vec<u32>,
    grad_b_bits: Vec<u32>,
}
impl Chain {
    fn layout(&self) -> DeepseekV41EngramLayout {
        assert_eq!(self.cols, 2);
        DeepseekV41EngramLayout {
            max_ngram_size: 2,
            layer_ids: vec![0],
            num_embeddings: vec![self.table_rows],
            primes: vec![vec![vec![2, 3]]],
            offsets: vec![vec![0, 2]],
            n_heads: 2,
            head_dim: self.head_dim,
        }
    }
    fn weights(&self) -> DeepseekV41EngramWeights {
        DeepseekV41EngramWeights {
            embedding: Vec::new(),
            wkv: Vec::new(),
            q_weight: values(&self.weight_bits),
            k_weight: vec![1.0; self.hc * self.hidden],
        }
    }
    fn table(&self) -> Result<DeepseekV41PackedWeights> {
        DeepseekV41PackedWeights::new(
            self.table_rows,
            self.head_dim,
            DeepseekV41WeightFormat::EngramFp8Row32,
            self.table.data.clone(),
            self.table.scales.clone(),
        )
    }
    fn native(&self, device: &VulkanDevice) -> Result<DeepseekV41Engram> {
        let mut weights = self.weights();
        if self.table.dtype == "dense" {
            weights.embedding = values(&self.table.values_bits);
            weights.wkv = values(&self.projection.values_bits);
            DeepseekV41Engram::new(
                device.clone(),
                &self.layout(),
                0,
                self.hc,
                self.hidden,
                self.eps,
                weights,
            )
        } else {
            DeepseekV41Engram::new_mixed(
                device.clone(),
                &self.layout(),
                0,
                self.hc,
                self.hidden,
                self.eps,
                weights,
                self.table()?,
                self.projection.native(true)?,
                precision(&self.precision),
            )
        }
    }
}
fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_engram_grad.json"
    ))
    .unwrap();
    assert_eq!(f.tolerance, 2e-7);
    assert_eq!(f.rtol, 0.0);
    assert_eq!(f.comparison, "strict_less_than");
    f
}
fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}
fn gpu(device: &VulkanDevice, bits: &[u32]) -> Result<GpuBuffer> {
    GpuBuffer::from_u32(device, bits)
}
fn precision(value: &str) -> DeepseekV41WeightOutput {
    if value == "bf16" {
        DeepseekV41WeightOutput::Bf16
    } else {
        DeepseekV41WeightOutput::Fp32
    }
}
fn gate(label: &str, actual: &[f32], expected: &[u32], failures: &mut Vec<String>) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut worst = 0.0f64;
    let mut at = 0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let b = f32::from_bits(b);
        assert!(a.is_finite() && b.is_finite(), "{label} non-finite at {i}");
        let drift = (a as f64 - b as f64).abs();
        if drift > worst {
            worst = drift;
            at = i;
        }
    }
    eprintln!(
        "V4.1 Engram adjoint {label}: {} values, worst {worst:.17e}",
        actual.len()
    );
    if worst >= 2e-7 {
        failures.push(format!(
            "{label}: {worst:.17e} at {at}, native={} cpu={}",
            actual[at],
            f32::from_bits(expected[at])
        ));
    }
}

#[test]
fn engram_normalized_gate_vjps_match_cpu_including_width5120() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41EngramAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().gates {
        let x = gpu(&device, &c.stream_bits)?;
        let kv = gpu(&device, &c.kv_bits)?;
        let weight = gpu(&device, &c.weight_bits)?;
        let mask = c.mask.as_ref().map(|v| gpu(&device, v)).transpose()?;
        let config = DeepseekV41EngramGateConfig {
            rows: c.rows,
            hc: c.hc,
            hidden: c.hidden,
            eps: c.eps,
            precision: precision(&c.precision),
        };
        let tape = adjoints.gate_forward(&x, &kv, &weight, mask.as_ref(), config)?;
        gate(
            &format!("{} forward", c.name),
            &tape.output.read_f32(c.output_bits.len())?,
            &c.output_bits,
            &mut failures,
        );
        if c.precision == "bf16" {
            assert_eq!(
                tape.output.read_u32(c.output_bits.len())?,
                c.output_bits,
                "{} BF16 forward bits",
                c.name
            );
        }
        x.write_f32(&vec![7.0; c.stream_bits.len()])?;
        kv.write_f32(&vec![3.0; c.kv_bits.len()])?;
        weight.write_f32(&vec![-1.0; c.weight_bits.len()])?;
        if let Some(mask) = mask {
            mask.write_u32(&vec![0; c.rows])?;
        }
        let gradient = adjoints.gate_backward(&tape, &gpu(&device, &c.upstream_bits)?)?;
        gate(
            &format!("{} dStream", c.name),
            &gradient.grad_stream.read_f32(c.grad_stream_bits.len())?,
            &c.grad_stream_bits,
            &mut failures,
        );
        gate(
            &format!("{} dKV", c.name),
            &gradient.grad_kv.read_f32(c.grad_kv_bits.len())?,
            &c.grad_kv_bits,
            &mut failures,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

fn check_chain(
    c: &Chain,
    g: &DeepseekV41EngramGradients,
    failures: &mut Vec<String>,
) -> Result<()> {
    for (label, actual, expected) in [
        (
            "dStream",
            g.grad_stream.read_f32(c.grad_stream_bits.len())?,
            &c.grad_stream_bits,
        ),
        (
            "dKV",
            g.grad_kv.read_f32(c.grad_kv_bits.len())?,
            &c.grad_kv_bits,
        ),
        (
            "dEmbedded",
            g.projection
                .grad_input
                .read_f32(c.grad_embedded_bits.len())?,
            &c.grad_embedded_bits,
        ),
    ] {
        gate(&format!("{} {label}", c.name), &actual, expected, failures);
    }
    if c.projection.rank != 0 {
        let gradients = g.projection.read_adapter_gradient("layers.0.engram.wkv")?;
        gate(
            &format!("{} dA", c.name),
            &gradients.grad_a,
            &c.grad_a_bits,
            failures,
        );
        gate(
            &format!("{} dB", c.name),
            &gradients.grad_b,
            &c.grad_b_bits,
            failures,
        );
    } else {
        assert!(g.projection.grad_a.is_none() && g.projection.grad_b.is_none());
    }
    Ok(())
}

#[test]
fn engram_retained_lookup_projection_chain_vjps_and_weight_release_match_cpu() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41EngramAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().chains {
        let engram = c.native(&device)?;
        let x = gpu(&device, &c.stream_bits)?;
        let hashes = gpu(&device, &c.hashes)?;
        let mask = gpu(&device, &c.mask)?;
        let inference = engram.apply(&x, &hashes, Some(&mask), c.rows)?;
        let tape = adjoints.forward(&engram, &x, &hashes, Some(&mask), c.rows)?;
        assert_eq!(
            tape.output.read_u32(c.output_bits.len())?,
            inference.read_u32(c.output_bits.len())?,
            "{} actual inference identity",
            c.name
        );
        gate(
            &format!("{} forward", c.name),
            &tape.output.read_f32(c.output_bits.len())?,
            &c.output_bits,
            &mut failures,
        );
        x.write_f32(&vec![7.0; c.stream_bits.len()])?;
        hashes.write_u32(&vec![u32::MAX; c.hashes.len()])?;
        mask.write_u32(&vec![0; c.rows])?;
        let before = device.memory_stats()?.live_buffer_bytes;
        drop(engram);
        let after = device.memory_stats()?.live_buffer_bytes;
        let frozen_table = if c.table.dtype == "dense" {
            c.table.values_bits.len() * 4
        } else {
            c.table.data.len() + c.table.scales.len()
        };
        let frozen_projection = if c.projection.dtype.starts_with("fp8") {
            c.projection.data.len() + c.projection.scales.len()
        } else {
            c.projection.values_bits.len() * 4
        };
        assert!(
            before - after >= frozen_table + frozen_projection,
            "{} tape pinned frozen table/projection",
            c.name
        );
        let base = c.projection.native(false)?.upload(&device)?;
        let gradient = adjoints.backward(&tape, &base, &gpu(&device, &c.upstream_bits)?)?;
        check_chain(&c, &gradient, &mut failures)?;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

struct Package(PathBuf);
impl Drop for Package {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
impl Package {
    fn new(c: &Chain) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "hierarchos-v41-engram-adjoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&path)?;
        let owner = Self(path);
        let mut config: serde_json::Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
        ))?;
        for (key, value) in [
            ("hidden_size", serde_json::json!(c.hidden)),
            ("hc_mult", serde_json::json!(c.hc)),
            ("engram_layer_ids", serde_json::json!([0])),
            ("engram_num_embeddings", serde_json::json!([c.table_rows])),
            ("engram_max_ngram_size", serde_json::json!(2)),
            ("engram_vocab_size", serde_json::json!(2)),
            ("engram_n_heads", serde_json::json!(c.cols)),
            ("engram_head_dim", serde_json::json!(c.head_dim)),
        ] {
            config["text_config"][key] = value;
        }
        fs::write(owner.0.join("config.json"), serde_json::to_vec(&config)?)?;
        let dense = c.table.dtype == "dense";
        let data: Vec<u8> = if dense {
            c.table
                .values_bits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        } else {
            c.table.data.clone()
        };
        let mut header = serde_json::Map::new();
        header.insert("layers.0.engram.embed.weight".into(),serde_json::json!({
            "dtype":if dense {"F32"} else {"F8_E4M3"},"shape":[c.table_rows,c.head_dim],"data_offsets":[0,data.len()]}));
        let mut payload = data;
        if !dense {
            let offset = payload.len();
            payload.extend_from_slice(&c.table.scales);
            header.insert("layers.0.engram.embed.scale".into(),serde_json::json!({
                "dtype":"F8_E8M0","shape":[c.table_rows,c.head_dim/32],"data_offsets":[offset,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&header)?;
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        fs::write(owner.0.join("model.safetensors"), file)?;
        Ok(owner)
    }
    fn streamed(&self, c: &Chain, device: &VulkanDevice) -> Result<DeepseekV41Engram> {
        let checkpoint = DeepseekV41Checkpoint::open(&self.0)?;
        let mut weights = c.weights();
        if c.table.dtype == "dense" {
            weights.wkv = values(&c.projection.values_bits);
            DeepseekV41Engram::new_streamed(
                device.clone(),
                &c.layout(),
                0,
                c.hc,
                c.hidden,
                c.eps,
                weights,
                checkpoint,
                0,
            )
        } else {
            DeepseekV41Engram::new_mixed_streamed(
                device.clone(),
                &c.layout(),
                0,
                c.hc,
                c.hidden,
                c.eps,
                weights,
                c.projection.native(true)?,
                precision(&c.precision),
                checkpoint,
                0,
            )
        }
    }
}

#[test]
fn engram_streamed_table_is_unneeded_by_retained_backward() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41EngramAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture()
        .chains
        .into_iter()
        .filter(|c| c.table.dtype == "dense" || c.projection.dtype == "fp8_block32")
    {
        let package = Package::new(&c)?;
        let engram = package.streamed(&c, &device)?;
        assert_eq!(engram.resident_table_count(), 0);
        let x = gpu(&device, &c.stream_bits)?;
        let hashes = gpu(&device, &c.hashes)?;
        let mask = gpu(&device, &c.mask)?;
        let tape = adjoints.forward(&engram, &x, &hashes, Some(&mask), c.rows)?;
        assert_eq!(
            tape.output.read_u32(c.output_bits.len())?,
            engram
                .apply(&x, &hashes, Some(&mask), c.rows)?
                .read_u32(c.output_bits.len())?
        );
        gate(
            &format!("{} streamed forward", c.name),
            &tape.output.read_f32(c.output_bits.len())?,
            &c.output_bits,
            &mut failures,
        );
        let source = package.0.join("model.safetensors");
        let parked = package.0.join("model.safetensors.unavailable");
        fs::rename(&source, &parked)?;
        assert!(
            engram.apply(&x, &hashes, Some(&mask), c.rows).is_err(),
            "forward must require the streamed table payload"
        );
        drop(engram);
        let base = c.projection.native(false)?.upload(&device)?;
        let gradient = adjoints.backward(&tape, &base, &gpu(&device, &c.upstream_bits)?)?;
        fs::rename(&parked, &source)?;
        check_chain(&c, &gradient, &mut failures)?;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn engram_adjoint_rejects_overflow_geometry_precision_and_foreign_buffers() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41EngramAdjoints::new(device.clone())?;
    let c = fixture().gates.remove(0);
    let x = gpu(&device, &c.stream_bits)?;
    let kv = gpu(&device, &c.kv_bits)?;
    let weight = gpu(&device, &c.weight_bits)?;
    let valid = DeepseekV41EngramGateConfig {
        rows: c.rows,
        hc: c.hc,
        hidden: c.hidden,
        eps: c.eps,
        precision: DeepseekV41WeightOutput::Fp32,
    };
    for config in [
        DeepseekV41EngramGateConfig { rows: 0, ..valid },
        DeepseekV41EngramGateConfig {
            hc: usize::MAX,
            ..valid
        },
        DeepseekV41EngramGateConfig {
            hidden: usize::MAX,
            ..valid
        },
        DeepseekV41EngramGateConfig { eps: 0.0, ..valid },
        DeepseekV41EngramGateConfig {
            eps: f32::NAN,
            ..valid
        },
        DeepseekV41EngramGateConfig {
            rows: 65_536,
            ..valid
        },
    ] {
        assert!(adjoints
            .gate_forward(&x, &kv, &weight, None, config)
            .is_err());
    }
    let short = GpuBuffer::zeros_f32(&device, 1)?;
    assert!(adjoints
        .gate_forward(&x, &short, &weight, None, valid)
        .is_err());
    assert!(adjoints
        .gate_forward(&x, &kv, &weight, Some(&short), valid)
        .is_err());
    let tape = adjoints.gate_forward(&x, &kv, &weight, None, valid)?;
    assert!(adjoints.gate_backward(&tape, &short).is_err());
    let foreign = VulkanDevice::new()?;
    assert!(adjoints
        .gate_forward(&gpu(&foreign, &c.stream_bits)?, &kv, &weight, None, valid)
        .is_err());
    assert!(adjoints
        .gate_backward(&tape, &gpu(&foreign, &c.upstream_bits)?)
        .is_err());
    Ok(())
}
