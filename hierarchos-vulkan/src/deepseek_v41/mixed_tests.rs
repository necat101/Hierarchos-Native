use super::*;
use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    revision: String,
    seed: u64,
    linears: Vec<Linear>,
    moes: Vec<Moe>,
    cast: Cast,
}
#[derive(Deserialize)]
struct Cast {
    input_bits: Vec<u32>,
    expected_bits: Vec<u32>,
}
#[derive(Clone, Deserialize)]
struct Weight {
    rows: usize,
    cols: usize,
    dtype: String,
    #[serde(default)]
    values: Vec<f32>,
    #[serde(default)]
    data: Vec<u8>,
    #[serde(default)]
    scales: Vec<u8>,
}
#[derive(Deserialize)]
struct Linear {
    name: String,
    rows: usize,
    precision: String,
    weight: Weight,
    input: Vec<f32>,
    expected_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    experts: usize,
    activated: usize,
    score_func: String,
    gate_temp: f32,
    norm_topk_prob: bool,
    route_scale: f32,
    swiglu_limit: f32,
}
impl Config {
    fn native(&self) -> DeepseekV41MoeConfig {
        DeepseekV41MoeConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            experts: self.experts,
            activated: self.activated,
            score_func: DeepseekV41MoeScoreFunc::parse(&self.score_func).unwrap(),
            gate_temp: self.gate_temp,
            norm_topk_prob: self.norm_topk_prob,
            route_scale: self.route_scale,
            swiglu_limit: self.swiglu_limit,
        }
    }
}
#[derive(Clone, Deserialize)]
struct Expert {
    w1: Weight,
    w2: Weight,
    w3: Weight,
}
#[derive(Deserialize)]
struct Moe {
    name: String,
    rows: usize,
    precision: String,
    config: Config,
    gate_weight: Vec<f32>,
    gate_bias: Vec<f32>,
    experts: Vec<Expert>,
    shared: Expert,
    input: Vec<f32>,
    expected_bits: Vec<u32>,
    expected_indices: Vec<u32>,
    expected_weights: Vec<f32>,
    counts: Vec<u32>,
}
fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_mixed.json"
    ))
    .unwrap();
    assert_eq!(f.revision, REFERENCE_REVISION);
    assert_eq!(f.seed, 4141040);
    assert_eq!(f.linears.len(), 5);
    assert_eq!(f.moes.len(), 4);
    f
}
fn precision(p: &str) -> DeepseekV41WeightOutput {
    match p {
        "fp32" => DeepseekV41WeightOutput::Fp32,
        "bf16" => DeepseekV41WeightOutput::Bf16,
        _ => panic!("unknown precision"),
    }
}
fn weight(w: Weight) -> Result<DeepseekV41LinearWeights> {
    Ok(match w.dtype.as_str() {
        "fp32" => DeepseekV41LinearWeights::Fp32 {
            rows: w.rows,
            cols: w.cols,
            values: w.values,
            lora: None,
        },
        "bf16" => DeepseekV41LinearWeights::Bf16 {
            rows: w.rows,
            cols: w.cols,
            values: w.values,
            lora: None,
        },
        "fp8_block32" | "fp4_row32" => DeepseekV41LinearWeights::Packed {
            weight: DeepseekV41PackedWeights::new(
                w.rows,
                w.cols,
                if w.dtype == "fp4_row32" {
                    DeepseekV41WeightFormat::Fp4Row32
                } else {
                    DeepseekV41WeightFormat::Fp8Block32
                },
                w.data,
                w.scales,
            )?,
            lora: None,
        },
        _ => panic!("unknown weight dtype"),
    })
}
fn expert(e: Expert) -> Result<DeepseekV41MixedExpertWeights> {
    Ok(DeepseekV41MixedExpertWeights {
        w1: weight(e.w1)?,
        w2: weight(e.w2)?,
        w3: weight(e.w3)?,
    })
}

#[derive(Deserialize)]
struct MixedAttentionFixture {
    seed: u64,
    revision: String,
    attentions: Vec<MixedAttentionCase>,
    scores: Vec<MixedScoreCase>,
}
#[derive(Deserialize)]
struct MixedAttentionCase {
    batch: usize,
    hidden: usize,
    q_rank: usize,
    heads: usize,
    dim: usize,
    groups: usize,
    output_rank: usize,
    precision: String,
    weights: std::collections::BTreeMap<String, Weight>,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    sink: Vec<f32>,
    steps: Vec<MixedAttentionStep>,
}
#[derive(Deserialize)]
struct MixedAttentionStep {
    pos: usize,
    seq: usize,
    input: Vec<f32>,
    query_bits: Vec<u32>,
    expected_bits: Vec<u32>,
    cache_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct MixedScoreCase {
    batch: usize,
    seq: usize,
    pos: usize,
    heads: usize,
    dim: usize,
    ratio: usize,
    width: usize,
    scale: f32,
    query: Vec<f32>,
    keys: Vec<f32>,
    weights: Vec<f32>,
    candidates: Vec<u32>,
    expected_bits: Vec<u32>,
}
fn mixed_attention_fixture() -> MixedAttentionFixture {
    let f: MixedAttentionFixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_mixed_attention.json"
    ))
    .unwrap();
    assert_eq!(f.seed, 4141050);
    assert_eq!(f.revision, REFERENCE_REVISION);
    assert_eq!(f.attentions.len(), 2);
    assert_eq!(f.scores.len(), 4);
    f
}
impl MixedAttentionCase {
    fn config(&self) -> DeepseekV41AttentionConfig {
        DeepseekV41AttentionConfig {
            heads: self.heads,
            head_dim: self.dim,
            hidden_size: self.hidden,
            q_lora_rank: self.q_rank,
            o_groups: self.groups,
            o_lora_rank: self.output_rank,
            window_size: 4,
            compress_ratio: 0,
            norm_eps: 1e-20,
            rotary: DeepseekV41RotaryConfig::sliding(16, 16, 10000.0),
        }
    }
    fn weights(&self) -> Result<DeepseekV41MixedAttentionWeights> {
        Ok(DeepseekV41MixedAttentionWeights {
            wq_a: weight(self.weights["wq_a"].clone())?,
            q_norm: self.q_norm.clone(),
            wq_b: weight(self.weights["wq_b"].clone())?,
            wkv: weight(self.weights["wkv"].clone())?,
            kv_norm: self.kv_norm.clone(),
            wo_a: weight(self.weights["wo_a"].clone())?,
            wo_b: weight(self.weights["wo_b"].clone())?,
            attn_sink: self.sink.clone(),
        })
    }
}
fn exact_bits(label: &str, actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len(), "{label} shape");
    assert!(!actual.is_empty(), "{label} empty comparison");
    if let Some((i, (a, b))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!(
            "{label} index {i}: native={a:08x}, CPU PyTorch={b:08x}; {} differences",
            actual.iter().zip(expected).filter(|(a, b)| a != b).count()
        );
    }
}

#[test]
fn deepseek_v41_mixed_attention_matches_cpu_bits_through_prefill_decode_reset() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    for c in mixed_attention_fixture().attentions {
        let mut native = DeepseekV41Attention::new_mixed(
            device.clone(),
            c.config(),
            c.weights()?,
            c.batch,
            precision(&c.precision),
        )?;
        for step in c.steps {
            let rows = c.batch * step.seq;
            let input = crate::GpuBuffer::from_f32(&device, &step.input)?;
            exact_bits(
                "query latent",
                &native
                    .query_latent(&input, rows)?
                    .read_u32(rows * c.q_rank)?,
                &step.query_bits,
            );
            let output = if step.pos == 0 {
                native.prefill(&input, None, None, c.batch, step.seq)?
            } else {
                native.decode_step(&input, None, None, c.batch, step.pos)?
            };
            exact_bits(
                "attention",
                &output.read_u32(rows * c.hidden)?,
                &step.expected_bits,
            );
            let cache = native
                .window_cache()
                .read_all()?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            exact_bits("window cache", &cache, &step.cache_bits);
        }
    }
    Ok(())
}

#[test]
fn deepseek_v41_bf16_index_scores_match_cpu_bits_including_masks_and_32_heads() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let scorer = DeepseekV41IndexScorer::new(device.clone())?;
    for c in mixed_attention_fixture().scores {
        let q = crate::GpuBuffer::from_f32(&device, &c.query)?;
        let k = crate::GpuBuffer::from_f32(&device, &c.keys)?;
        let w = crate::GpuBuffer::from_f32(&device, &c.weights)?;
        let mask = crate::GpuBuffer::from_u32(&device, &c.candidates)?;
        let scores = scorer.score_typed(
            &q,
            &k,
            &w,
            Some(&mask),
            c.batch * c.seq,
            c.batch,
            c.seq,
            c.heads,
            c.dim,
            c.width,
            c.width,
            c.pos,
            c.ratio,
            c.scale,
            DeepseekV41WeightOutput::Bf16,
            DeepseekV41WeightOutput::Bf16,
        )?;
        exact_bits(
            "BF16 index scores",
            &scores.read_u32(c.expected_bits.len())?,
            &c.expected_bits,
        );
    }
    Ok(())
}

#[test]
fn deepseek_v41_mixed_attention_rejects_unconverted_grouped_weights_and_invalid_storage(
) -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let c = mixed_attention_fixture().attentions.remove(0);
    let mut w = c.weights()?;
    let [rows, cols] = w.wo_a.shape();
    w.wo_a = DeepseekV41LinearWeights::Packed {
        weight: DeepseekV41PackedWeights::new(
            rows,
            cols,
            DeepseekV41WeightFormat::Fp8Block32,
            vec![0; rows * cols],
            vec![127; rows.div_ceil(32) * (cols / 32)],
        )?,
        lora: None,
    };
    assert!(DeepseekV41Attention::new_mixed(
        device.clone(),
        c.config(),
        w,
        c.batch,
        precision(&c.precision)
    )
    .err()
    .unwrap()
    .to_string()
    .contains("converted dense"));
    let mut w = c.weights()?;
    if let DeepseekV41LinearWeights::Bf16 { values, .. } = &mut w.wo_a {
        values[0] = 0.1;
    }
    assert!(DeepseekV41Attention::new_mixed(
        device.clone(),
        c.config(),
        w,
        c.batch,
        precision(&c.precision)
    )
    .err()
    .unwrap()
    .to_string()
    .contains("non-BF16"));
    let mut w = c.weights()?;
    w.q_norm.pop();
    assert!(DeepseekV41Attention::new_mixed(
        device,
        c.config(),
        w,
        c.batch,
        precision(&c.precision)
    )
    .is_err());
    Ok(())
}
fn moe_weights(c: &Moe) -> Result<DeepseekV41MixedMoeWeights> {
    Ok(DeepseekV41MixedMoeWeights {
        gate_weight: c.gate_weight.clone(),
        gate_bias: c.gate_bias.clone(),
        experts: c
            .experts
            .iter()
            .cloned()
            .map(expert)
            .collect::<Result<_>>()?,
        shared: expert(c.shared.clone())?,
    })
}

#[test]
fn deepseek_v41_mixed_linear_matches_typed_cpu_boundaries() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let kernel = DeepseekV41MixedLinearKernel::new(device.clone())?;
    let mut count = 0;
    for c in fixture().linears {
        let input = crate::GpuBuffer::from_f32(&device, &c.input)?;
        let w = weight(c.weight)?.upload(&device)?;
        let actual = kernel
            .apply(&input, c.rows, &w, precision(&c.precision))?
            .read_u32(c.expected_bits.len())?;
        assert_eq!(actual.len(), c.rows * w.shape()[0]);
        for (index, (&a, &b)) in actual.iter().zip(&c.expected_bits).enumerate() {
            let drift = (f32::from_bits(a) as f64 - f32::from_bits(b) as f64).abs();
            if c.precision == "bf16" {
                assert_eq!(a, b, "{} index {index}", c.name);
            } else {
                assert!(drift < 2e-7, "{} index {index}, {drift:e}", c.name);
            }
        }
        count += actual.len();
    }
    assert!(count > 0);
    eprintln!("V4.1 typed linear: {count} CPU-reference output values");
    Ok(())
}

#[test]
fn deepseek_v41_mixed_moe_matches_released_expert_and_moe_forward() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let mut count = 0;
    for c in fixture().moes {
        let module = DeepseekV41MixedMoe::new(
            device.clone(),
            c.config.native(),
            moe_weights(&c)?,
            precision(&c.precision),
        )?;
        let x = crate::GpuBuffer::from_f32(&device, &c.input)?;
        let actual = module.apply(&x, c.rows)?.read_u32(c.expected_bits.len())?;
        assert_eq!(actual.len(), c.rows * c.config.hidden_size);
        for (index, (&a, &b)) in actual.iter().zip(&c.expected_bits).enumerate() {
            assert_eq!(
                a,
                b,
                "{} output {index}: {} vs {}",
                c.name,
                f32::from_bits(a),
                f32::from_bits(b)
            );
        }
        let trace = module.route_trace.borrow();
        let (weights, ids, counts) = trace.as_ref().unwrap();
        assert_eq!(
            ids.read_u32(c.expected_indices.len())?,
            c.expected_indices,
            "{} routing",
            c.name
        );
        assert_eq!(
            counts.read_u32(c.counts.len())?,
            c.counts,
            "{} counts",
            c.name
        );
        for (a, b) in weights
            .read_f32(c.expected_weights.len())?
            .iter()
            .zip(&c.expected_weights)
        {
            assert!((*a as f64 - *b as f64).abs() < 2e-7);
        }
        eprintln!(
            "V4.1 mixed MoE {}: {} bit-exact outputs, counts {:?}",
            c.name,
            actual.len(),
            c.counts
        );
        count += actual.len();
    }
    assert!(count > 0);
    Ok(())
}

#[test]
fn deepseek_v41_mixed_storage_and_precision_validation() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let foreign = crate::VulkanDevice::new()?;
    let kernel = DeepseekV41MixedLinearKernel::new(device.clone())?;
    let x = crate::GpuBuffer::from_f32(&device, &[0.1; 32])?;
    let remote = crate::GpuBuffer::from_f32(&foreign, &[0.1; 32])?;
    let w = DeepseekV41LinearWeights::Bf16 {
        rows: 1,
        cols: 32,
        values: vec![0.5; 32],
        lora: None,
    }
    .upload(&device)?;
    assert!(kernel
        .apply(&x, 1, &w, DeepseekV41WeightOutput::Fp32)
        .is_err());
    assert!(kernel
        .apply(&remote, 1, &w, DeepseekV41WeightOutput::Bf16)
        .is_err());
    for (rows, cols, values) in [
        (0, 32, vec![]),
        (1, 32, vec![0.1; 32]),
        (1, 32, vec![f32::NAN; 32]),
        (1, 32, vec![0.5; 31]),
    ] {
        assert!(DeepseekV41LinearWeights::Bf16 {
            rows,
            cols,
            values,
            lora: None,
        }
        .upload(&device)
        .is_err());
    }
    let f = fixture();
    let c = &f.moes[0];
    let module = DeepseekV41MixedMoe::new(
        device.clone(),
        c.config.native(),
        moe_weights(c)?,
        DeepseekV41WeightOutput::Bf16,
    )?;
    assert!(module.apply(&remote, 1).is_err());
    assert!(module.apply(&x, 0).is_err());
    assert!(module.apply(&x, 65_536).is_err());
    Ok(())
}

#[test]
fn deepseek_v41_mixed_bf16_cast_matches_cpu_ties_subnormals_and_nonfinite_classes() -> Result<()> {
    let c = fixture().cast;
    let device = crate::VulkanDevice::new()?;
    let kernel = DeepseekV41MixedLinearKernel::new(device.clone())?;
    let input = crate::GpuBuffer::from_u32(&device, &c.input_bits)?;
    assert_eq!(
        kernel
            .cast_buffer(&input, c.input_bits.len(), DeepseekV41WeightOutput::Fp32)?
            .read_u32(c.input_bits.len())?,
        c.input_bits
    );
    let actual = kernel
        .cast_buffer(&input, c.input_bits.len(), DeepseekV41WeightOutput::Bf16)?
        .read_u32(c.input_bits.len())?;
    assert_eq!(actual.len(), c.expected_bits.len());
    for (&a, &b) in actual.iter().zip(&c.expected_bits) {
        if f32::from_bits(b).is_nan() {
            assert!(f32::from_bits(a).is_nan());
        } else {
            assert_eq!(a, b);
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct EngramBase {
    name: String,
    rows: usize,
    hc: usize,
    hidden: usize,
    head_dim: usize,
    table_rows: usize,
    cols: usize,
    data: Vec<u8>,
    scales: Vec<u8>,
    stream: Vec<f32>,
    hashes: Vec<i64>,
    mask: Vec<u32>,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
}
#[derive(Deserialize)]
struct MixedEngram {
    base: EngramBase,
    projection: Weight,
    expected_bits: Vec<u32>,
    expected_kv_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct EngramFixture {
    revision: String,
    cases: Vec<MixedEngram>,
}
fn engram_fixture() -> EngramFixture {
    let f: EngramFixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_mixed_engram.json"
    ))
    .unwrap();
    assert_eq!(f.revision, REFERENCE_REVISION);
    assert_eq!(f.cases.len(), 2);
    f
}
impl EngramBase {
    fn weights(&self) -> DeepseekV41EngramWeights {
        DeepseekV41EngramWeights {
            embedding: vec![],
            wkv: vec![],
            q_weight: self.q_weight.clone(),
            k_weight: self.k_weight.clone(),
        }
    }
    fn table(&self) -> Result<DeepseekV41PackedWeights> {
        DeepseekV41PackedWeights::new(
            self.table_rows,
            self.head_dim,
            DeepseekV41WeightFormat::EngramFp8Row32,
            self.data.clone(),
            self.scales.clone(),
        )
    }
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
}

#[test]
fn deepseek_v41_mixed_engram_matches_released_bf16_table_projection_and_residual() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let mut count = 0;
    for c in engram_fixture().cases {
        let b = c.base;
        let engram = DeepseekV41Engram::new_mixed(
            device.clone(),
            &b.layout(),
            0,
            b.hc,
            b.hidden,
            1e-20,
            b.weights(),
            b.table()?,
            weight(c.projection)?,
            DeepseekV41WeightOutput::Bf16,
        )?;
        let input = crate::GpuBuffer::from_f32(&device, &b.stream)?;
        let hashes = crate::GpuBuffer::from_u32(
            &device,
            &b.hashes.iter().map(|&v| v as u32).collect::<Vec<_>>(),
        )?;
        let mask = crate::GpuBuffer::from_u32(&device, &b.mask)?;
        let actual = engram
            .apply(&input, &hashes, Some(&mask), b.rows)?
            .read_u32(c.expected_bits.len())?;
        assert_eq!(actual.len(), b.rows * b.hc * b.hidden);
        assert_eq!(
            engram
                .projection_trace()
                .unwrap()
                .read_u32(c.expected_kv_bits.len())?,
            c.expected_kv_bits,
            "{} projection",
            b.name
        );
        assert_eq!(actual, c.expected_bits, "{} mixed residual", b.name);
        count += actual.len();
    }
    assert_eq!(count, 2224);
    eprintln!("V4.1 mixed Engram: {count} bit-exact BF16 residuals, including HC=4");
    Ok(())
}

#[test]
fn deepseek_v41_mixed_engram_checkpoint_preserves_projection_storage_and_stream_policy(
) -> Result<()> {
    use safetensors::{
        serialize_to_file,
        tensor::{Dtype, TensorView},
    };
    use serde_json::json;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    struct Temp(PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            if let Ok(entries) = fs::read_dir(&self.0) {
                for e in entries.flatten() {
                    let _ = fs::remove_file(e.path());
                }
            }
            let _ = fs::remove_dir(&self.0);
        }
    }
    static ID: AtomicU64 = AtomicU64::new(0);
    let p = Temp(std::env::temp_dir().join(format!(
        "hierarchos-v41-mixed-engram-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    )));
    fs::create_dir(&p.0)?;
    let c = engram_fixture().cases.remove(0);
    let b = &c.base;
    let mut config: serde_json::Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
    ))?;
    for (key, value) in [
        ("vocab_size", json!(16)),
        ("dspark_block_size", json!(0)),
        ("hidden_size", json!(b.hidden)),
        ("hc_mult", json!(b.hc)),
        ("engram_layer_ids", json!([0])),
        ("engram_num_embeddings", json!([b.table_rows])),
        ("engram_max_ngram_size", json!(2)),
        ("engram_n_heads", json!(b.cols)),
        ("engram_head_dim", json!(b.head_dim)),
        ("engram_vocab_size", json!(3)),
        ("engram_pad_token_id", json!(2)),
        ("engram_compressed_vocab_size", json!(8)),
    ] {
        config["text_config"][key] = value;
    }
    fs::write(p.0.join("config.json"), serde_json::to_vec(&config)?)?;
    serialize_to_file(
        vec![
            (
                "layers.0.engram.embed.weight",
                TensorView::new(Dtype::F8_E4M3, vec![b.table_rows, b.head_dim], &b.data)?,
            ),
            (
                "layers.0.engram.embed.scale",
                TensorView::new(
                    Dtype::F8_E8M0,
                    vec![b.table_rows, b.head_dim / 32],
                    &b.scales,
                )?,
            ),
            (
                "layers.0.engram.wkv.weight",
                TensorView::new(
                    Dtype::F8_E4M3,
                    vec![c.projection.rows, c.projection.cols],
                    &c.projection.data,
                )?,
            ),
            (
                "layers.0.engram.wkv.scale",
                TensorView::new(
                    Dtype::F8_E8M0,
                    vec![c.projection.rows.div_ceil(32), c.projection.cols / 32],
                    &c.projection.scales,
                )?,
            ),
            (
                "layers.0.engram.q_weight",
                TensorView::new(
                    Dtype::F32,
                    vec![b.hc, b.hidden],
                    bytemuck::cast_slice(&b.q_weight),
                )?,
            ),
            (
                "layers.0.engram.k_weight",
                TensorView::new(
                    Dtype::F32,
                    vec![b.hc, b.hidden],
                    bytemuck::cast_slice(&b.k_weight),
                )?,
            ),
        ],
        None,
        &p.0.join("model.safetensors"),
    )?;
    let device = crate::VulkanDevice::new()?;
    let mut checkpoint = DeepseekV41Checkpoint::open(&p.0)?;
    assert!(
        checkpoint.load_packed_engram(0).is_err(),
        "the old FP32 projection loader must reject packed projections"
    );
    let tokens = DeepseekV41EngramTokens {
        token_map: (0..16).map(|i| i % 8).collect(),
        multipliers: vec![vec![3, 5]],
        pad_token_id: 2,
    };
    let layout = DeepseekV41EngramLayout::from_text(&checkpoint.config.text)?.unwrap();
    let mut loaded = DeepseekV41EngramRuntime::from_checkpoint(
        device.clone(),
        &mut checkpoint,
        tokens.clone(),
        1,
        16,
    )?;
    let mut hash = DeepseekV41EngramHash::new(device.clone(), layout.clone(), tokens, 16, 1, 16)?;
    let (table, projection, weights) = checkpoint.load_mixed_engram(0)?;
    let direct = DeepseekV41Engram::new_mixed(
        device.clone(),
        &layout,
        0,
        b.hc,
        b.hidden,
        1e-20,
        weights,
        table,
        projection,
        DeepseekV41WeightOutput::Fp32,
    )?;
    for (pos, ids, mask) in [
        (0, vec![0, 1, 2], vec![1, 0, 1]),
        (3, vec![3], vec![1]),
        (0, vec![4], vec![1]),
    ] {
        let rows = ids.len();
        let ids = crate::GpuBuffer::from_u32(&device, &ids)?;
        let mask = crate::GpuBuffer::from_u32(&device, &mask)?;
        loaded.begin_step(&ids, pos, rows, Some(&mask))?;
        hash.begin_step(&ids, pos, rows, Some(&mask))?;
        let stream = crate::GpuBuffer::from_f32(&device, &b.stream[..rows * b.hc * b.hidden])?;
        let expected = direct
            .apply(&stream, &hash.hashes(0)?, Some(&mask), rows)?
            .read_u32(rows * b.hc * b.hidden)?;
        assert_eq!(
            loaded
                .apply(0, &stream, rows)?
                .unwrap()
                .read_u32(expected.len())?,
            expected
        );
    }
    Ok(())
}
