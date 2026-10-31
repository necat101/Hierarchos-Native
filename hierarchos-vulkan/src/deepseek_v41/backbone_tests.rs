use super::indexer::NormKernel;
use super::*;
use crate::{GpuBuffer, VulkanDevice};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Fixture {
    revision: String,
    hc_cases: Vec<HcCase>,
    route_cases: Vec<RouteCase>,
    moe_cases: Vec<MoeCase>,
}

#[derive(Deserialize)]
struct HcCase {
    name: String,
    hc: usize,
    dim: usize,
    rows: usize,
    sinkhorn_iters: usize,
    eps: f32,
    norm_eps: f32,
    atol: f64,
    stream: Vec<f32>,
    attn_fn: Vec<f32>,
    attn_base: Vec<f32>,
    attn_scale: Vec<f32>,
    sublayer: Vec<f32>,
    expected_pre: Vec<f32>,
    expected_post: Vec<f32>,
    expected_comb: Vec<f32>,
    expected_collapsed: Vec<f32>,
    expected_expanded: Vec<f32>,
}

#[derive(Deserialize)]
struct RouteCase {
    name: String,
    rows: usize,
    hidden: usize,
    experts: usize,
    topk: usize,
    score: String,
    gate_temp: f32,
    norm_topk: bool,
    route_scale: f32,
    atol: f64,
    x: Vec<f32>,
    gate_weight: Vec<f32>,
    gate_bias: Vec<f32>,
    expected_indices: Vec<i32>,
    expected_weights: Vec<f32>,
}

#[derive(Deserialize)]
struct MoeCase {
    name: String,
    rows: usize,
    hidden: usize,
    inter: usize,
    experts: usize,
    topk: usize,
    score: String,
    gate_temp: f32,
    norm_topk: bool,
    route_scale: f32,
    swiglu_limit: f32,
    atol: f64,
    x: Vec<f32>,
    gate_weight: Vec<f32>,
    gate_bias: Vec<f32>,
    expert_w1: Vec<Vec<f32>>,
    expert_w2: Vec<Vec<f32>>,
    expert_w3: Vec<Vec<f32>>,
    shared_w1: Vec<f32>,
    shared_w2: Vec<f32>,
    shared_w3: Vec<f32>,
    expected_indices: Vec<i32>,
    expected_weights: Vec<f32>,
    expected_output: Vec<f32>,
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_backbone.json"
    ))
    .unwrap();
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    fixture
}

fn max_drift(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "comparison length");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (*a as f64 - *e as f64).abs())
        .fold(0.0f64, f64::max)
}

fn moe_config(case: &MoeCase) -> DeepseekV41MoeConfig {
    DeepseekV41MoeConfig {
        hidden_size: case.hidden,
        intermediate_size: case.inter,
        experts: case.experts,
        activated: case.topk,
        score_func: DeepseekV41MoeScoreFunc::parse(&case.score).unwrap(),
        gate_temp: case.gate_temp,
        norm_topk_prob: case.norm_topk,
        route_scale: case.route_scale,
        swiglu_limit: case.swiglu_limit,
    }
}

#[test]
fn deepseek_v41_hyper_connections_match_the_pinned_reference_fixture() -> Result<()> {
    let fixture = fixture();
    let device = crate::VulkanDevice::new()?;
    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for case in &fixture.hc_cases {
        let mix_hc = (2 + case.hc) * case.hc;
        let width = case.hc * case.dim;
        let zeros = vec![0.0f32; mix_hc * width];
        let hyper = DeepseekV41HyperConnections::new(
            device.clone(),
            case.dim,
            case.hc,
            case.sinkhorn_iters,
            case.eps,
            case.norm_eps,
            DeepseekV41HyperConnectionWeights {
                attn_fn: case.attn_fn.clone(),
                attn_base: case.attn_base.clone(),
                attn_scale: case.attn_scale.clone(),
                ffn_fn: zeros,
                ffn_base: vec![0.0; mix_hc],
                ffn_scale: vec![0.0; 3],
            },
        )?;
        let stream = GpuBuffer::from_f32(&device, &case.stream)?;
        let mixes = hyper.attention_mixes(&stream, case.rows)?;
        let pre = mixes.pre.read_f32(case.rows * case.hc)?;
        let post = mixes.post.read_f32(case.rows * case.hc)?;
        let comb = mixes.comb.read_f32(case.rows * case.hc * case.hc)?;
        for (label, actual, expected) in [
            ("pre", &pre, &case.expected_pre),
            ("post", &post, &case.expected_post),
            ("comb", &comb, &case.expected_comb),
        ] {
            let drift = max_drift(actual, expected);
            assert!(
                drift <= case.atol,
                "{} {label} drift {drift:e} exceeds {}",
                case.name,
                case.atol
            );
            worst = worst.max(drift);
            compared += actual.len();
        }
        let collapsed = hyper.pre(&stream, &mixes.pre, case.rows)?;
        let collapsed = collapsed.read_f32(case.rows * case.dim)?;
        let expanded = hyper.post(
            &GpuBuffer::from_f32(&device, &case.sublayer)?,
            &stream,
            &mixes.post,
            &mixes.comb,
            case.rows,
        )?;
        let expanded = expanded.read_f32(case.rows * case.hc * case.dim)?;
        for (label, actual, expected) in [
            ("collapsed", &collapsed, &case.expected_collapsed),
            ("expanded", &expanded, &case.expected_expanded),
        ] {
            let drift = max_drift(actual, expected);
            assert!(
                drift <= case.atol,
                "{} {label} drift {drift:e} exceeds {}",
                case.name,
                case.atol
            );
            worst = worst.max(drift);
            compared += actual.len();
        }
    }
    assert!(compared > 0, "the coefficient comparison must be nonempty");
    eprintln!("hyper-connection worst drift {worst:e} over {compared} values");
    Ok(())
}

#[test]
fn deepseek_v41_moe_routing_matches_the_pinned_reference_fixture() -> Result<()> {
    let fixture = fixture();
    let device = crate::VulkanDevice::new()?;
    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for case in &fixture.route_cases {
        let moe_case = MoeCase {
            name: case.name.clone(),
            rows: case.rows,
            hidden: case.hidden,
            inter: 8,
            experts: case.experts,
            topk: case.topk,
            score: case.score.clone(),
            gate_temp: case.gate_temp,
            norm_topk: case.norm_topk,
            route_scale: case.route_scale,
            swiglu_limit: 0.0,
            atol: case.atol,
            x: case.x.clone(),
            gate_weight: case.gate_weight.clone(),
            gate_bias: case.gate_bias.clone(),
            expert_w1: vec![],
            expert_w2: vec![],
            expert_w3: vec![],
            shared_w1: vec![],
            shared_w2: vec![],
            shared_w3: vec![],
            expected_indices: case.expected_indices.clone(),
            expected_weights: case.expected_weights.clone(),
            expected_output: vec![],
        };
        let width = case.hidden * 8;
        let moe = DeepseekV41Moe::new(
            device.clone(),
            moe_config(&moe_case),
            DeepseekV41MoeWeights {
                gate_weight: case.gate_weight.clone(),
                gate_bias: case.gate_bias.clone(),
                expert_w1: vec![vec![0.0; width]; case.experts],
                expert_w2: vec![vec![0.0; width]; case.experts],
                expert_w3: vec![vec![0.0; width]; case.experts],
                shared_w1: vec![0.0; width],
                shared_w2: vec![0.0; width],
                shared_w3: vec![0.0; width],
            },
        )?;
        let x = GpuBuffer::from_f32(&device, &case.x)?;
        let (weights, indices) = moe.route(&x, case.rows)?;
        let indices = indices.read_u32(case.rows * case.topk)?;
        let expected: Vec<u32> = case
            .expected_indices
            .iter()
            .map(|value| *value as u32)
            .collect();
        assert_eq!(indices, expected, "{} routed indices", case.name);
        let weights = weights.read_f32(case.rows * case.topk)?;
        assert!(weights.iter().any(|&value| value != 0.0));
        let drift = max_drift(&weights, &case.expected_weights);
        assert!(
            drift <= case.atol,
            "{} routing weight drift {drift:e} exceeds {}",
            case.name,
            case.atol
        );
        worst = worst.max(drift);
        compared += weights.len();
    }
    assert!(compared > 0, "the comparison must exercise routing weights");
    eprintln!("MoE routing worst drift {worst:e} over {compared} values");
    Ok(())
}

#[test]
fn deepseek_v41_moe_matches_the_pinned_reference_fixture() -> Result<()> {
    let fixture = fixture();
    let device = crate::VulkanDevice::new()?;
    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for case in &fixture.moe_cases {
        let moe = DeepseekV41Moe::new(
            device.clone(),
            moe_config(case),
            DeepseekV41MoeWeights {
                gate_weight: case.gate_weight.clone(),
                gate_bias: case.gate_bias.clone(),
                expert_w1: case.expert_w1.clone(),
                expert_w2: case.expert_w2.clone(),
                expert_w3: case.expert_w3.clone(),
                shared_w1: case.shared_w1.clone(),
                shared_w2: case.shared_w2.clone(),
                shared_w3: case.shared_w3.clone(),
            },
        )?;
        let x = GpuBuffer::from_f32(&device, &case.x)?;
        let (weights, indices) = moe.route(&x, case.rows)?;
        let indices = indices.read_u32(case.rows * case.topk)?;
        let expected: Vec<u32> = case
            .expected_indices
            .iter()
            .map(|value| *value as u32)
            .collect();
        assert_eq!(indices, expected, "{} routed indices", case.name);
        let weights = weights.read_f32(case.rows * case.topk)?;
        let drift = max_drift(&weights, &case.expected_weights);
        assert!(
            drift <= case.atol,
            "{} routing weight drift {drift:e} exceeds {}",
            case.name,
            case.atol
        );
        worst = worst.max(drift);
        compared += weights.len();

        let output = moe.apply(&x, case.rows)?;
        let output = output.read_f32(case.rows * case.hidden)?;
        let drift = max_drift(&output, &case.expected_output);
        assert!(
            drift <= case.atol,
            "{} output drift {drift:e} exceeds {}",
            case.name,
            case.atol
        );
        worst = worst.max(drift);
        compared += output.len();
    }
    assert!(compared > 0, "the expert comparison must be nonempty");
    eprintln!("MoE worst drift {worst:e} over {compared} values");
    Ok(())
}

struct WiringConfig {
    config: DeepseekV41Config,
    runtime: DeepseekV41AttentionRuntimeConfig,
}

fn wiring_config() -> Result<WiringConfig> {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
    ))?;
    let patch = json!({
        "hidden_size":32, "head_dim":32, "qk_rope_head_dim":16,
        "q_lora_rank":8, "num_attention_heads":2, "o_groups":1, "o_lora_rank":16,
        "index_n_heads":2, "index_head_dim":16, "index_topk":2,
        "sliding_window":4, "max_position_embeddings":16,
        "num_hidden_layers":2, "num_nextn_predict_layers":0,
        "compress_ratios":[0,0], "kv_source_layer_ids":[],
        "index_source_layer_ids":[], "candidate_source_layer_id":-1,
        "candidate_block_size":0, "candidate_topk_blocks":0,
        "hc_mult":2, "hc_sinkhorn_iters":20, "hc_eps":1e-6,
        "moe_intermediate_size":12, "n_routed_experts":2, "n_shared_experts":1,
        "num_experts_per_tok":1, "scoring_func":"sqrtsoftplus",
        "norm_topk_prob":true, "routed_scaling_factor":1.3, "swiglu_limit":1.5,
        "engram_layer_ids":[], "engram_num_embeddings":[],
        "dspark_block_size":0, "dspark_target_layer_ids":[],
        "rope_scaling":null
    });
    value["text_config"]
        .as_object_mut()
        .unwrap()
        .extend(patch.as_object().unwrap().clone());
    Ok(WiringConfig {
        config: DeepseekV41Config::from_hf_value(&value)?,
        runtime: DeepseekV41AttentionRuntimeConfig {
            batch_size: 2,
            max_prefill: 8,
            max_positions: 16,
        },
    })
}

struct Lcg(u64);

impl Lcg {
    fn vector(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((self.0 >> 33) as f32 / 2_147_483_648.0 - 0.5) * scale
            })
            .collect()
    }
}

struct WiringWeights {
    attention: DeepseekV41AttentionWeights,
    hyper: DeepseekV41HyperConnectionWeights,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    moe: DeepseekV41MoeWeights,
}

fn wiring_weights(layer: usize) -> WiringWeights {
    let mut rng = Lcg(9001 + layer as u64 * 17);
    let hidden = 32;
    let mix_hc = 8;
    WiringWeights {
        attention: DeepseekV41AttentionWeights {
            wq_a: rng.vector(8 * hidden, 0.2),
            q_norm: vec![1.05; 8],
            wq_b: rng.vector(2 * 32 * 8, 0.15),
            wkv: rng.vector(32 * hidden, 0.15),
            kv_norm: vec![0.95; 32],
            wo_a: rng.vector(16 * 64, 0.12),
            wo_b: rng.vector(hidden * 16, 0.12),
            attn_sink: rng.vector(2, 0.2),
        },
        hyper: DeepseekV41HyperConnectionWeights {
            attn_fn: rng.vector(mix_hc * 2 * hidden, 0.02),
            attn_base: rng.vector(mix_hc, 0.1),
            attn_scale: vec![0.8, 1.1, 0.9],
            ffn_fn: rng.vector(mix_hc * 2 * hidden, 0.02),
            ffn_base: rng.vector(mix_hc, 0.1),
            ffn_scale: vec![1.0, 0.9, 1.2],
        },
        attn_norm: rng.vector(hidden, 0.3).iter().map(|v| 1.0 + v).collect(),
        ffn_norm: rng.vector(hidden, 0.3).iter().map(|v| 1.0 + v).collect(),
        moe: DeepseekV41MoeWeights {
            gate_weight: rng.vector(2 * hidden, 0.15),
            gate_bias: rng.vector(2, 0.2),
            expert_w1: (0..2).map(|_| rng.vector(12 * hidden, 0.12)).collect(),
            expert_w2: (0..2).map(|_| rng.vector(hidden * 12, 0.12)).collect(),
            expert_w3: (0..2).map(|_| rng.vector(12 * hidden, 0.2)).collect(),
            shared_w1: rng.vector(12 * hidden, 0.12),
            shared_w2: rng.vector(hidden * 12, 0.12),
            shared_w3: rng.vector(12 * hidden, 0.2),
        },
    }
}

fn attention_config(
    ratio: usize,
    runtime: DeepseekV41AttentionRuntimeConfig,
) -> DeepseekV41AttentionConfig {
    DeepseekV41AttentionConfig {
        heads: 2,
        head_dim: 32,
        hidden_size: 32,
        q_lora_rank: 8,
        o_groups: 1,
        o_lora_rank: 16,
        window_size: 4,
        compress_ratio: ratio,
        norm_eps: 1e-20,
        rotary: DeepseekV41RotaryConfig::sliding(16, runtime.max_positions, 10_000.0),
    }
}

/// Explicit released block order, built from standalone components.
struct ManualDriver {
    device: VulkanDevice,
    attention: Vec<DeepseekV41Attention>,
    hyper: Vec<DeepseekV41HyperConnections>,
    moe: Vec<DeepseekV41Moe>,
    attn_norm: Vec<GpuBuffer>,
    ffn_norm: Vec<GpuBuffer>,
    norm: NormKernel,
    hidden: usize,
    eps: f32,
    engram: Option<DeepseekV41EngramRuntime>,
}

impl ManualDriver {
    fn new(
        device: &VulkanDevice,
        runtime: DeepseekV41AttentionRuntimeConfig,
        weights: Vec<WiringWeights>,
        engram: Option<DeepseekV41EngramRuntime>,
    ) -> Result<Self> {
        let moe_config = DeepseekV41MoeConfig {
            hidden_size: 32,
            intermediate_size: 12,
            experts: 2,
            activated: 1,
            score_func: DeepseekV41MoeScoreFunc::SqrtSoftplus,
            gate_temp: 1.0,
            norm_topk_prob: true,
            route_scale: 1.3,
            swiglu_limit: 1.5,
        };
        let mut result = Self {
            device: device.clone(),
            attention: vec![],
            hyper: vec![],
            moe: vec![],
            attn_norm: vec![],
            ffn_norm: vec![],
            norm: NormKernel::new(device)?,
            hidden: 32,
            eps: 1e-20,
            engram,
        };
        for w in weights {
            result.attention.push(DeepseekV41Attention::new(
                device.clone(),
                attention_config(0, runtime),
                w.attention,
                runtime.batch_size,
            )?);
            result.hyper.push(DeepseekV41HyperConnections::new(
                device.clone(),
                32,
                2,
                20,
                1e-6,
                1e-20,
                w.hyper,
            )?);
            result
                .attn_norm
                .push(GpuBuffer::from_f32(device, &w.attn_norm)?);
            result
                .ffn_norm
                .push(GpuBuffer::from_f32(device, &w.ffn_norm)?);
            result
                .moe
                .push(DeepseekV41Moe::new(device.clone(), moe_config, w.moe)?);
        }
        Ok(result)
    }

    fn run(
        &mut self,
        stream: &GpuBuffer,
        seq: usize,
        decode: bool,
        start_pos: usize,
        token_ids: Option<&GpuBuffer>,
        token_mask: Option<&GpuBuffer>,
    ) -> Result<GpuBuffer> {
        let rows = 2 * seq;
        if let Some(engram) = &mut self.engram {
            engram.begin_step(token_ids.expect("tokens"), start_pos, seq, token_mask)?;
        }
        let mut current = stream.clone();
        let mut pre_mix = self.hyper[0].identity_pre_mix(rows)?;
        for layer in 0..self.hyper.len() {
            if let Some(engram) = &self.engram {
                if let Some(injected) = engram.apply(layer, &current, rows)? {
                    current = injected;
                }
            }
            let attn = self.hyper[layer].attention_mixes(&current, rows)?;
            let collapsed = self.hyper[layer].pre(&current, &pre_mix, rows)?;
            let collapsed = self.norm.apply(
                &self.device,
                &collapsed,
                &self.attn_norm[layer],
                rows,
                self.hidden,
                self.eps,
            )?;
            let attended = if decode {
                self.attention[layer].decode_step(&collapsed, None, None, 2, start_pos)?
            } else {
                self.attention[layer].prefill(&collapsed, None, None, 2, seq)?
            };
            current = self.hyper[layer].post(&attended, &current, &attn.post, &attn.comb, rows)?;
            let ffn = self.hyper[layer].ffn_mixes(&current, rows)?;
            let collapsed = self.hyper[layer].pre(&current, &attn.pre, rows)?;
            let collapsed = self.norm.apply(
                &self.device,
                &collapsed,
                &self.ffn_norm[layer],
                rows,
                self.hidden,
                self.eps,
            )?;
            let mixed = self.moe[layer].apply(&collapsed, rows)?;
            current = self.hyper[layer].post(&mixed, &current, &ffn.post, &ffn.comb, rows)?;
            pre_mix = ffn.pre;
        }
        self.hyper[0].pre(&current, &pre_mix, rows)
    }
}

#[test]
fn deepseek_v41_backbone_runs_the_released_block_order() -> Result<()> {
    let wiring = wiring_config()?;
    let device = crate::VulkanDevice::new()?;
    let backbone_weights: Vec<_> = (0..2)
        .map(|layer| {
            let w = wiring_weights(layer);
            DeepseekV41BackboneLayerWeights {
                attention: DeepseekV41AttentionLayerWeights {
                    attention: w.attention,
                    compressor: None,
                    indexer: None,
                },
                hyper: w.hyper,
                attn_norm: w.attn_norm,
                ffn_norm: w.ffn_norm,
                moe: w.moe,
            }
        })
        .collect();
    let mut backbone = DeepseekV41Backbone::new(
        device.clone(),
        &wiring.config,
        wiring.runtime,
        backbone_weights,
        None,
    )?;
    let mut manual = ManualDriver::new(
        &device,
        wiring.runtime,
        (0..2).map(wiring_weights).collect(),
        None,
    )?;

    let mut rng = Lcg(777);
    let prefill = GpuBuffer::from_f32(&device, &rng.vector(2 * 4 * 2 * 32, 0.5))?;
    let native = backbone.prefill(&prefill, 4, None, None)?;
    let expected = manual.run(&prefill, 4, false, 0, None, None)?;
    let native_values = native.read_f32(8 * 32)?;
    let expected_values = expected.read_f32(8 * 32)?;
    assert_eq!(
        native_values, expected_values,
        "prefill must follow the explicit block order bit-for-bit"
    );
    assert!(native_values.iter().all(|value| value.is_finite()));
    assert!(native_values.iter().any(|value| *value != 0.0));
    assert_eq!(backbone.next_position(), Some(4));
    assert_eq!(backbone.next_layer(), None);

    let decode = GpuBuffer::from_f32(&device, &rng.vector(2 * 2 * 32, 0.5))?;
    let native = backbone.decode(&decode, 4, None, None)?;
    let expected = manual.run(&decode, 1, true, 4, None, None)?;
    assert_eq!(
        native.read_f32(2 * 32)?,
        expected.read_f32(2 * 32)?,
        "decode must continue the same Hyper-Connection/attention/MoE order"
    );
    assert_eq!(backbone.next_position(), Some(5));
    assert!(backbone.decode(&decode, 9, None, None).is_err());
    Ok(())
}

fn engram_weights(layer: usize) -> DeepseekV41EngramWeights {
    let mut rng = Lcg(7001 + layer as u64 * 13);
    DeepseekV41EngramWeights {
        embedding: rng.vector(512 * 4, 0.5),
        wkv: rng.vector(96 * 8, 0.2),
        q_weight: rng.vector(2 * 32, 0.3),
        k_weight: rng.vector(2 * 32, 0.3),
    }
}

fn engram_runtime(
    device: &VulkanDevice,
    config: &DeepseekV41Config,
) -> Result<DeepseekV41EngramRuntime> {
    let layout = DeepseekV41EngramLayout::from_text(&config.text)?.expect("engram layout");
    DeepseekV41EngramRuntime::new(
        device.clone(),
        layout,
        DeepseekV41EngramTokens {
            token_map: (0..16).map(|value| value % 8).collect(),
            multipliers: vec![vec![3, 5], vec![7, 9]],
            pad_token_id: 2,
        },
        (0..2).map(|layer| Some(engram_weights(layer))).collect(),
        16,
        2,
        2,
        32,
        1e-20,
        2,
        16,
    )
}

#[test]
fn deepseek_v41_backbone_applies_engram_before_scheduled_blocks() -> Result<()> {
    let mut wiring = wiring_config()?;
    wiring.config.text.vocab_size = 16;
    wiring.config.text.engram_layer_ids = vec![0, 1];
    wiring.config.text.engram_num_embeddings = vec![512, 512];
    wiring.config.text.engram_max_ngram_size = 2;
    wiring.config.text.engram_vocab_size = 50;
    wiring.config.text.engram_n_heads = 2;
    wiring.config.text.engram_head_dim = 4;
    wiring.config.text.engram_pad_token_id = 2;
    wiring.config.text.engram_compressed_vocab_size = 8;

    let device = crate::VulkanDevice::new()?;
    let backbone_weights: Vec<_> = (0..2)
        .map(|layer| {
            let w = wiring_weights(layer);
            DeepseekV41BackboneLayerWeights {
                attention: DeepseekV41AttentionLayerWeights {
                    attention: w.attention,
                    compressor: None,
                    indexer: None,
                },
                hyper: w.hyper,
                attn_norm: w.attn_norm,
                ffn_norm: w.ffn_norm,
                moe: w.moe,
            }
        })
        .collect();
    let runtime = engram_runtime(&device, &wiring.config)?;
    let manual_runtime = engram_runtime(&device, &wiring.config)?;
    let mut backbone = DeepseekV41Backbone::new(
        device.clone(),
        &wiring.config,
        wiring.runtime,
        backbone_weights,
        Some(runtime),
    )?;
    let mut manual = ManualDriver::new(
        &device,
        wiring.runtime,
        (0..2).map(wiring_weights).collect(),
        Some(manual_runtime),
    )?;

    let mut rng = Lcg(909);
    let prefill = GpuBuffer::from_f32(&device, &rng.vector(2 * 4 * 2 * 32, 0.5))?;
    let tokens = GpuBuffer::from_u32(&device, &[0u32, 1, 2, 3, 4, 5, 6, 7])?;
    let native = backbone.prefill(&prefill, 4, Some(&tokens), None)?;
    let expected = manual.run(&prefill, 4, false, 0, Some(&tokens), None)?;
    assert_eq!(
        native.read_f32(8 * 32)?,
        expected.read_f32(8 * 32)?,
        "prefill must inject Engram before each scheduled block"
    );
    assert_eq!(backbone.next_position(), Some(4));

    let decode = GpuBuffer::from_f32(&device, &rng.vector(2 * 2 * 32, 0.5))?;
    let decode_tokens = GpuBuffer::from_u32(&device, &[8u32, 9])?;
    let native = backbone.decode(&decode, 4, Some(&decode_tokens), None)?;
    let expected = manual.run(&decode, 1, true, 4, Some(&decode_tokens), None)?;
    assert_eq!(
        native.read_f32(2 * 32)?,
        expected.read_f32(2 * 32)?,
        "decode must reuse the persistent hash cache"
    );
    Ok(())
}
