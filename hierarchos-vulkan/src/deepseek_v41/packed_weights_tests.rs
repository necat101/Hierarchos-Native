use super::*;
use anyhow::Result;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use serde::Deserialize;
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Deserialize)]
struct Fixture {
    revision: String,
    seed: u64,
    model_sha256: String,
    kernel_sha256: String,
    converter_sha256: String,
    decodes: Vec<Decode>,
    linears: Vec<Linear>,
    engrams: Vec<EngramCase>,
}
#[derive(Deserialize)]
struct Decode {
    name: String,
    rows: usize,
    cols: usize,
    format: String,
    data: Vec<u8>,
    scales: Vec<u8>,
    expected_fp32_bits: Vec<u32>,
    expected_bf16_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Linear {
    name: String,
    rows: usize,
    outputs: usize,
    inner: usize,
    format: String,
    act_block: usize,
    data: Vec<u8>,
    scales: Vec<u8>,
    input: Vec<f32>,
    expected_fp32_bits: Vec<u32>,
    expected_bf16_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct EngramCase {
    name: String,
    rows: usize,
    hc: usize,
    hidden: usize,
    head_dim: usize,
    table_rows: usize,
    cols: usize,
    data: Vec<u8>,
    scales: Vec<u8>,
    table: Vec<f32>,
    stream: Vec<f32>,
    hashes: Vec<i64>,
    mask: Vec<u32>,
    wkv: Vec<f32>,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    expected_lookup_bits: Vec<u32>,
    expected: Vec<f32>,
}
impl EngramCase {
    fn layout(&self) -> DeepseekV41EngramLayout {
        DeepseekV41EngramLayout {
            max_ngram_size: 2,
            layer_ids: vec![0],
            num_embeddings: vec![self.table_rows],
            primes: vec![vec![vec![3; self.cols]]],
            offsets: vec![vec![0; self.cols]],
            n_heads: self.cols,
            head_dim: self.head_dim,
        }
    }
    fn weights(&self, dense: bool) -> DeepseekV41EngramWeights {
        DeepseekV41EngramWeights {
            embedding: if dense { self.table.clone() } else { vec![] },
            wkv: self.wkv.clone(),
            q_weight: self.q_weight.clone(),
            k_weight: self.k_weight.clone(),
        }
    }
    fn packed(&self) -> Result<DeepseekV41PackedWeights> {
        DeepseekV41PackedWeights::new(
            self.table_rows,
            self.head_dim,
            DeepseekV41WeightFormat::EngramFp8Row32,
            self.data.clone(),
            self.scales.clone(),
        )
    }
}
fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_packed_weights.json"
    ))
    .unwrap();
    assert_eq!(f.revision, REFERENCE_REVISION);
    assert_eq!(f.seed, 4141030);
    assert_eq!(
        f.model_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    assert_eq!(
        f.kernel_sha256,
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(
        f.converter_sha256,
        "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"
    );
    assert_eq!(f.decodes.len(), 6);
    assert_eq!(f.linears.len(), 8);
    assert_eq!(f.engrams.len(), 2);
    f
}
fn format(s: &str) -> DeepseekV41WeightFormat {
    match s {
        "fp8_block32" => DeepseekV41WeightFormat::Fp8Block32,
        "fp8_block128" => DeepseekV41WeightFormat::Fp8Block128,
        "fp4_row32" => DeepseekV41WeightFormat::Fp4Row32,
        "engram_fp8_row32" => DeepseekV41WeightFormat::EngramFp8Row32,
        _ => panic!("unknown fixture format {s}"),
    }
}
fn weights(c: &Decode) -> Result<DeepseekV41PackedWeights> {
    DeepseekV41PackedWeights::new(
        c.rows,
        c.cols,
        format(&c.format),
        c.data.clone(),
        c.scales.clone(),
    )
}

#[test]
fn deepseek_v41_packed_decoding_matches_independent_cpu_bit_exactly() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let kernel = DeepseekV41PackedWeightKernel::new(device.clone())?;
    let mut compared = 0;
    for c in fixture().decodes {
        let resident = weights(&c)?.upload(&device)?;
        for (output, expected) in [
            (DeepseekV41WeightOutput::Fp32, &c.expected_fp32_bits),
            (DeepseekV41WeightOutput::Bf16, &c.expected_bf16_bits),
        ] {
            assert_eq!(expected.len(), c.rows * c.cols);
            let actual = kernel
                .dequantize(&resident, output)?
                .read_u32(expected.len())?;
            for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    a, b,
                    "{} {output:?}, index {index}: {a:08x} vs {b:08x}",
                    c.name
                );
            }
            compared += actual.len();
        }
    }
    assert!(compared > 0);
    eprintln!("V4.1 packed decode: {compared} bit-exact FP32/BF16 values");
    Ok(())
}

#[test]
fn deepseek_v41_packed_linear_matches_independent_cpu_blocks() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let kernel = DeepseekV41PackedWeightKernel::new(device.clone())?;
    let mut compared = 0;
    let mut worst = 0.0f64;
    let mut failures = vec![];
    for c in fixture().linears {
        let w =
            DeepseekV41PackedWeights::new(c.outputs, c.inner, format(&c.format), c.data, c.scales)?
                .upload(&device)?;
        assert_eq!(c.input.len(), c.rows * c.inner);
        let input = crate::GpuBuffer::from_f32(&device, &c.input)?;
        for (output, expected) in [
            (DeepseekV41WeightOutput::Fp32, &c.expected_fp32_bits),
            (DeepseekV41WeightOutput::Bf16, &c.expected_bf16_bits),
        ] {
            assert_eq!(expected.len(), c.rows * c.outputs);
            let actual = kernel
                .linear_buffer(&input, c.rows, &w, c.act_block, output)?
                .read_u32(expected.len())?;
            let mut case_worst = 0.0f64;
            for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                let (a_value, b_value) = (f32::from_bits(a), f32::from_bits(b));
                assert!(a_value.is_finite() && b_value.is_finite());
                let drift = (a_value as f64 - b_value as f64).abs();
                case_worst = case_worst.max(drift);
                if (output == DeepseekV41WeightOutput::Bf16 && a != b)
                    || (output == DeepseekV41WeightOutput::Fp32 && drift >= 2e-7)
                {
                    failures.push(format!(
                        "{} {output:?} {index}: {a_value} vs {b_value} ({drift:.17e})",
                        c.name
                    ));
                }
            }
            worst = worst.max(case_worst);
            eprintln!(
                "V4.1 packed linear {} {output:?}: {case_worst:.17e}",
                c.name
            );
            compared += actual.len();
        }
    }
    assert!(compared > 0);
    assert!(failures.is_empty(), "packed linear failures: {failures:?}");
    eprintln!("V4.1 packed linear: {compared} values, worst drift {worst:.17e}; CPU component mirror only");
    Ok(())
}

struct Package {
    root: PathBuf,
}
impl Package {
    fn new() -> Result<Self> {
        static ID: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hierarchos-v41-packed-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        fs::write(
            root.join("config.json"),
            include_bytes!("../../validation/fixtures/deepseek_v41_flash_1017_config.json"),
        )?;
        Ok(Self { root })
    }
    fn write(
        &self,
        filename: &str,
        name: &str,
        dtype: Dtype,
        shape: Vec<usize>,
        data: &[u8],
    ) -> Result<()> {
        serialize_to_file(
            vec![(name, TensorView::new(dtype, shape, data)?)],
            None,
            &self.root.join(filename),
        )?;
        Ok(())
    }
}
impl Drop for Package {
    fn drop(&mut self) {
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let _ = fs::remove_dir(&self.root);
    }
}

#[test]
fn deepseek_v41_packed_checkpoint_reads_exported_and_converted_sharded_payloads() -> Result<()> {
    for c in fixture().decodes {
        let fmt = format(&c.format);
        let fp4 = fmt == DeepseekV41WeightFormat::Fp4Row32;
        for exported in [false, true] {
            let p = Package::new()?;
            let (name, scale) = if exported {
                (
                    "model.layers.0.mlp.experts.0.w1.weight",
                    "model.layers.0.mlp.experts.0.w1.weight_scale_inv",
                )
            } else {
                (
                    "layers.0.ffn.experts.0.w1.weight",
                    "layers.0.ffn.experts.0.w1.scale",
                )
            };
            let dtype = if fp4 {
                if exported {
                    Dtype::I8
                } else {
                    Dtype::F4
                }
            } else {
                Dtype::F8_E4M3
            };
            let shape = vec![c.rows, if fp4 && exported { c.cols / 2 } else { c.cols }];
            p.write("weight.safetensors", name, dtype, shape, &c.data)?;
            p.write(
                "scale.safetensors",
                scale,
                Dtype::F8_E8M0,
                fmt.scale_shape(c.rows, c.cols).to_vec(),
                &c.scales,
            )?;
            fs::write(
                p.root.join("model.safetensors.index.json"),
                serde_json::to_vec(
                    &json!({"weight_map":{name:"weight.safetensors",scale:"scale.safetensors"}}),
                )?,
            )?;
            let mut reader = DeepseekV41Checkpoint::open(&p.root)?;
            let actual = reader.read_packed_weight(name, scale, c.rows, c.cols, fmt)?;
            assert_eq!(actual.rows(), c.rows);
            assert_eq!(actual.cols(), c.cols);
            assert_eq!(actual.format(), fmt);
            assert_eq!(actual.data(), c.data);
            assert_eq!(actual.scales(), c.scales);
            assert!(reader.read_tensor(name, &[c.rows, c.cols]).is_err());
        }
    }
    Ok(())
}

#[test]
fn deepseek_v41_packed_checkpoint_rejects_invalid_dtype_shapes_scales_and_offsets() -> Result<()> {
    let p = Package::new()?;
    let fmt = DeepseekV41WeightFormat::Fp8Block32;
    for (weight_dtype, weight_shape, scale_dtype, scale_shape, byte, scale) in [
        ("U8", vec![1, 32], "F8_E8M0", vec![1, 1], 56, 127),
        ("F8_E4M3", vec![32, 1], "F8_E8M0", vec![1, 1], 56, 127),
        ("F8_E4M3", vec![1, 32], "F8_E4M3", vec![1, 1], 56, 127),
        ("F8_E4M3", vec![1, 32], "F8_E8M0", vec![1], 56, 127),
        ("F8_E4M3", vec![1, 32], "F8_E8M0", vec![1, 1], 127, 127),
        ("F8_E4M3", vec![1, 32], "F8_E8M0", vec![1, 1], 56, 255),
    ] {
        let header = serde_json::to_vec(
            &json!({"w":{"dtype":weight_dtype,"shape":weight_shape,"data_offsets":[0,32]},"s":{"dtype":scale_dtype,"shape":scale_shape,"data_offsets":[32,33]}}),
        )?;
        let mut raw = (header.len() as u64).to_le_bytes().to_vec();
        raw.extend(header);
        raw.extend(vec![byte; 32]);
        raw.push(scale);
        fs::write(p.root.join("model.safetensors"), raw)?;
        let mut r = DeepseekV41Checkpoint::open(&p.root)?;
        assert!(r.read_packed_weight("w", "s", 1, 32, fmt).is_err());
    }
    for offsets in [[0, 33], [5, 4], [u64::MAX - 32, u64::MAX]] {
        let header = serde_json::to_vec(
            &json!({"w":{"dtype":"F8_E4M3","shape":[1,32],"data_offsets":offsets},"s":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[32,33]}}),
        )?;
        let mut raw = (header.len() as u64).to_le_bytes().to_vec();
        raw.extend(header);
        raw.extend(vec![56; 32]);
        raw.push(127);
        fs::write(p.root.join("model.safetensors"), raw)?;
        let mut r = DeepseekV41Checkpoint::open(&p.root)?;
        assert!(r.read_packed_weight("w", "s", 1, 32, fmt).is_err());
    }
    Ok(())
}

#[test]
fn deepseek_v41_packed_weights_validate_geometry_and_finite_materialization() -> Result<()> {
    let fmt = DeepseekV41WeightFormat::Fp8Block32;
    for (rows, cols) in [(0, 32), (1, 0), (1, 31), (65_536, 32), (usize::MAX, 32)] {
        assert!(DeepseekV41PackedWeights::new(rows, cols, fmt, vec![], vec![]).is_err());
    }
    for (data, scales) in [
        (vec![56; 31], vec![127]),
        (vec![56; 32], vec![]),
        (vec![127; 32], vec![127]),
        (vec![255; 32], vec![127]),
        (vec![56; 32], vec![255]),
        (vec![126; 32], vec![254]),
    ] {
        assert!(DeepseekV41PackedWeights::new(1, 32, fmt, data, scales).is_err());
    }
    Ok(())
}

#[test]
fn deepseek_v41_packed_kernel_rejects_foreign_buffers_and_invalid_precision_geometry() -> Result<()>
{
    let device = crate::VulkanDevice::new()?;
    let foreign = crate::VulkanDevice::new()?;
    let kernel = DeepseekV41PackedWeightKernel::new(device.clone())?;
    let w = DeepseekV41PackedWeights::new(
        1,
        128,
        DeepseekV41WeightFormat::Fp8Block32,
        vec![56; 128],
        vec![127; 4],
    )?;
    let resident = w.upload(&device)?;
    let remote = w.upload(&foreign)?;
    let input = crate::GpuBuffer::from_f32(&device, &[1.0; 128])?;
    let remote_input = crate::GpuBuffer::from_f32(&foreign, &[1.0; 128])?;
    assert!(kernel
        .dequantize(&remote, DeepseekV41WeightOutput::Fp32)
        .is_err());
    assert!(kernel
        .linear_buffer(
            &remote_input,
            1,
            &resident,
            32,
            DeepseekV41WeightOutput::Bf16
        )
        .is_err());
    for (rows, block) in [(0, 32), (65_536, 32), (1, 16), (1, 128), (2, 32)] {
        assert!(kernel
            .linear_buffer(
                &input,
                rows,
                &resident,
                block,
                DeepseekV41WeightOutput::Bf16
            )
            .is_err());
    }
    let engram = DeepseekV41PackedWeights::new(
        1,
        128,
        DeepseekV41WeightFormat::EngramFp8Row32,
        vec![56; 128],
        vec![127; 4],
    )?
    .upload(&device)?;
    assert!(kernel
        .linear_buffer(&input, 1, &engram, 32, DeepseekV41WeightOutput::Bf16)
        .is_err());
    Ok(())
}

#[test]
fn deepseek_v41_packed_engram_lookup_and_residual_match_released_forward() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let (mut lookups, mut residuals, mut worst) = (0, 0, 0.0f64);
    let mut traces = vec![];
    for c in fixture().engrams {
        let layout = c.layout();
        let packed = DeepseekV41Engram::new_packed(
            device.clone(),
            &layout,
            0,
            c.hc,
            c.hidden,
            1e-20,
            c.weights(false),
            c.packed()?,
        )?;
        let dense = DeepseekV41Engram::new(
            device.clone(),
            &layout,
            0,
            c.hc,
            c.hidden,
            1e-20,
            c.weights(true),
        )?;
        let hashes = crate::GpuBuffer::from_u32(
            &device,
            &c.hashes.iter().map(|&v| v as u32).collect::<Vec<_>>(),
        )?;
        let result = packed
            .lookup_rows(&hashes, c.rows * c.cols)?
            .read_u32(c.expected_lookup_bits.len())?;
        assert_eq!(result, c.expected_lookup_bits, "{} lookup", c.name);
        lookups += result.len();
        let stream = crate::GpuBuffer::from_f32(&device, &c.stream)?;
        let mask = crate::GpuBuffer::from_u32(&device, &c.mask)?;
        let result = packed
            .apply(&stream, &hashes, Some(&mask), c.rows)?
            .read_f32(c.expected.len())?;
        let baseline = dense
            .apply(&stream, &hashes, Some(&mask), c.rows)?
            .read_u32(c.expected.len())?;
        traces.push(json!({"name":c.name,"kv":packed.projection_trace().unwrap().read_f32(c.rows*(c.hc+1)*c.hidden)?,"output":result}));
        assert_eq!(
            result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            baseline,
            "{} packed/dense residual",
            c.name
        );
        for (&a, &b) in result.iter().zip(&c.expected) {
            assert!(a.is_finite() && b.is_finite());
            let drift = (a as f64 - b as f64).abs();
            // Existing Engram component ceiling, separate from the unchanged
            // strict full-model gate. Packed/dense paths must be bit-exact.
            assert!(drift <= 2e-6, "{} residual drift {drift:e}", c.name);
            worst = worst.max(drift);
        }
        residuals += result.len();
        assert!(DeepseekV41Engram::new_packed(
            device.clone(),
            &layout,
            0,
            c.hc,
            c.hidden,
            1e-20,
            c.weights(true),
            c.packed()?
        )
        .is_err());
        let wrong = DeepseekV41PackedWeights::new(
            c.table_rows,
            c.head_dim,
            DeepseekV41WeightFormat::Fp8Block32,
            c.data.clone(),
            vec![127; c.table_rows.div_ceil(32) * (c.head_dim / 32)],
        )?;
        assert!(DeepseekV41Engram::new_packed(
            device.clone(),
            &layout,
            0,
            c.hc,
            c.hidden,
            1e-20,
            c.weights(false),
            wrong
        )
        .is_err());
    }
    assert!(lookups > 0 && residuals > 0);
    eprintln!("V4.1 packed Engram: {lookups} exact lookup values, {residuals} residuals, worst CPU drift {worst:.17e}; packed/dense outputs bit-exact");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/packed-engram-native-trace.json");
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, serde_json::to_vec(&json!({"cases":traces}))?)?;
    Ok(())
}

#[test]
fn deepseek_v41_packed_engram_checkpoint_runtime_preserves_cache_and_reset() -> Result<()> {
    let f = fixture();
    let c = &f.engrams[0];
    let p = Package::new()?;
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(p.root.join("config.json"))?)?;
    for (key, value) in [
        ("vocab_size", json!(16)),
        ("dspark_block_size", json!(0)),
        ("hidden_size", json!(c.hidden)),
        ("hc_mult", json!(c.hc)),
        ("engram_layer_ids", json!([0])),
        ("engram_num_embeddings", json!([c.table_rows])),
        ("engram_max_ngram_size", json!(2)),
        ("engram_n_heads", json!(c.cols)),
        ("engram_head_dim", json!(c.head_dim)),
        ("engram_vocab_size", json!(3)),
        ("engram_pad_token_id", json!(2)),
        ("engram_compressed_vocab_size", json!(8)),
    ] {
        config["text_config"][key] = value;
    }
    fs::write(p.root.join("config.json"), serde_json::to_vec(&config)?)?;
    serialize_to_file(
        vec![
            (
                "layers.0.engram.embed.weight",
                TensorView::new(Dtype::F8_E4M3, vec![c.table_rows, c.head_dim], &c.data)?,
            ),
            (
                "layers.0.engram.embed.scale",
                TensorView::new(
                    Dtype::F8_E8M0,
                    vec![c.table_rows, c.head_dim / 32],
                    &c.scales,
                )?,
            ),
            (
                "layers.0.engram.wkv.weight",
                TensorView::new(
                    Dtype::F32,
                    vec![(c.hc + 1) * c.hidden, c.cols * c.head_dim],
                    bytemuck::cast_slice(&c.wkv),
                )?,
            ),
            (
                "layers.0.engram.q_weight",
                TensorView::new(
                    Dtype::F32,
                    vec![c.hc, c.hidden],
                    bytemuck::cast_slice(&c.q_weight),
                )?,
            ),
            (
                "layers.0.engram.k_weight",
                TensorView::new(
                    Dtype::F32,
                    vec![c.hc, c.hidden],
                    bytemuck::cast_slice(&c.k_weight),
                )?,
            ),
        ],
        None,
        &p.root.join("model.safetensors"),
    )?;
    let device = crate::VulkanDevice::new()?;
    let mut reader = DeepseekV41Checkpoint::open(&p.root)?;
    let tokens = DeepseekV41EngramTokens {
        token_map: (0..16).map(|i| i % 8).collect(),
        multipliers: vec![vec![3, 5]],
        pad_token_id: 2,
    };
    let mut packed = DeepseekV41EngramRuntime::from_checkpoint(
        device.clone(),
        &mut reader,
        tokens.clone(),
        1,
        16,
    )?;
    let text = &reader.config.text;
    let layout = DeepseekV41EngramLayout::from_text(text)?.unwrap();
    let mut weights: Vec<_> = (0..text.num_hidden_layers).map(|_| None).collect();
    weights[0] = Some(c.weights(true));
    let mut dense = DeepseekV41EngramRuntime::new(
        device.clone(),
        layout,
        tokens,
        weights,
        16,
        text.num_hidden_layers,
        c.hc,
        c.hidden,
        1e-20,
        1,
        16,
    )?;
    for (pos, ids, mask) in [
        (0, vec![0, 1, 2], vec![1, 0, 1]),
        (3, vec![3], vec![1]),
        (0, vec![4], vec![1]),
        (1, vec![5], vec![0]),
    ] {
        let rows = ids.len();
        let token_ids = crate::GpuBuffer::from_u32(&device, &ids)?;
        let token_mask = crate::GpuBuffer::from_u32(&device, &mask)?;
        packed.begin_step(&token_ids, pos, rows, Some(&token_mask))?;
        dense.begin_step(&token_ids, pos, rows, Some(&token_mask))?;
        let stream = crate::GpuBuffer::from_f32(&device, &c.stream[..rows * c.hc * c.hidden])?;
        assert_eq!(
            packed
                .apply(0, &stream, rows)?
                .unwrap()
                .read_u32(rows * c.hc * c.hidden)?,
            dense
                .apply(0, &stream, rows)?
                .unwrap()
                .read_u32(rows * c.hc * c.hidden)?
        );
    }
    assert!(reader.load_packed_engram(1).is_err());
    p.write(
        "model.safetensors",
        "layers.0.engram.embed.weight",
        Dtype::F8_E4M3,
        vec![c.table_rows, c.head_dim],
        &c.data,
    )?;
    let mut missing = DeepseekV41Checkpoint::open(&p.root)?;
    assert!(missing.load_packed_engram(0).is_err());
    Ok(())
}

#[test]
fn deepseek_v41_packed_decode_tiles_large_row_tables_without_expanding_weights() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let kernel = DeepseekV41PackedWeightKernel::new(device.clone())?;
    let (rows, cols) = (65_536, 64);
    let w = DeepseekV41PackedWeights::new(
        rows,
        cols,
        DeepseekV41WeightFormat::EngramFp8Row32,
        vec![56; rows * cols],
        vec![127; rows * (cols / 32)],
    )?
    .upload(&device)?;
    let result = kernel
        .dequantize(&w, DeepseekV41WeightOutput::Fp32)?
        .read_u32(rows * cols)?;
    assert_eq!(result.len(), rows * cols);
    assert!(result.iter().all(|&b| b == 1.0f32.to_bits()));
    Ok(())
}
