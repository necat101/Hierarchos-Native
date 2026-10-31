use super::*;
use crate::deepseek_v41::{
    peft::DeepseekV41LoraHost, DeepseekV41LoraAdapter, DeepseekV41PackedWeights,
    DeepseekV41WeightFormat,
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf, sync::Arc};

#[derive(Deserialize)]
struct Fixture {
    tolerance: f64,
    comparison: String,
    rtol: f64,
    cases: Vec<Case>,
    swiglu: Vec<Swiglu>,
    fp32_clamp: Fp32Clamp,
}
#[derive(Clone, Deserialize)]
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
    fn native(&self) -> Result<DeepseekV41MoeConfig> {
        Ok(DeepseekV41MoeConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            experts: self.experts,
            activated: self.activated,
            score_func: DeepseekV41MoeScoreFunc::parse(&self.score_func)?,
            gate_temp: self.gate_temp,
            norm_topk_prob: self.norm_topk_prob,
            route_scale: self.route_scale,
            swiglu_limit: self.swiglu_limit,
        })
    }
}
#[derive(Clone, Deserialize)]
struct Projection {
    rows: usize,
    cols: usize,
    dtype: String,
    data: Vec<u8>,
    scales: Vec<u8>,
    rank: usize,
    scale: f32,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
impl Projection {
    fn format(&self) -> DeepseekV41WeightFormat {
        match self.dtype.as_str() {
            "fp8_block32" => DeepseekV41WeightFormat::Fp8Block32,
            "fp8_block128" => DeepseekV41WeightFormat::Fp8Block128,
            "fp4_row32" => DeepseekV41WeightFormat::Fp4Row32,
            _ => panic!("invalid fixture format"),
        }
    }
    fn native(&self, adapters: bool) -> Result<DeepseekV41LinearWeights> {
        Ok(DeepseekV41LinearWeights::Packed {
            weight: DeepseekV41PackedWeights::new(
                self.rows,
                self.cols,
                self.format(),
                self.data.clone(),
                self.scales.clone(),
            )?,
            lora: (adapters && self.rank != 0).then(|| DeepseekV41LoraHost {
                rank: self.rank,
                scale: self.scale,
                a: values(&self.a_bits).into(),
                b: values(&self.b_bits).into(),
            }),
        })
    }
}
#[derive(Clone, Deserialize)]
struct Expert {
    w1: Projection,
    w2: Projection,
    w3: Projection,
}
impl Expert {
    fn native(&self, adapters: bool) -> Result<DeepseekV41MixedExpertWeights> {
        Ok(DeepseekV41MixedExpertWeights {
            w1: self.w1.native(adapters)?,
            w2: self.w2.native(adapters)?,
            w3: self.w3.native(adapters)?,
        })
    }
}
#[derive(Clone, Deserialize)]
struct FactorGradient {
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
#[derive(Clone, Deserialize)]
struct DownTrace {
    grad_input_bits: Vec<u32>,
}
#[derive(Clone, Deserialize)]
struct Case {
    name: String,
    rows: usize,
    config: Config,
    precision: String,
    gate_weight: Vec<f32>,
    gate_bias: Vec<f32>,
    experts: Vec<Expert>,
    shared: Expert,
    x_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    output_bits: Vec<u32>,
    indices: Vec<u32>,
    route_bits: Vec<u32>,
    counts: Vec<u32>,
    grad_x_bits: Vec<u32>,
    grad_route_bits: Vec<u32>,
    gradients: BTreeMap<String, Option<FactorGradient>>,
    trace: BTreeMap<String, DownTrace>,
}
impl Case {
    fn precision(&self) -> DeepseekV41WeightOutput {
        if self.precision == "bf16" {
            DeepseekV41WeightOutput::Bf16
        } else {
            DeepseekV41WeightOutput::Fp32
        }
    }
    fn native(&self, device: &VulkanDevice, adapters: bool) -> Result<DeepseekV41MixedMoe> {
        DeepseekV41MixedMoe::new(
            device.clone(),
            self.config.native()?,
            DeepseekV41MixedMoeWeights {
                gate_weight: self.gate_weight.clone(),
                gate_bias: self.gate_bias.clone(),
                experts: self
                    .experts
                    .iter()
                    .map(|e| e.native(adapters))
                    .collect::<Result<_>>()?,
                shared: self.shared.native(adapters)?,
            },
            self.precision(),
        )
    }
}
#[derive(Deserialize)]
struct Swiglu {
    name: String,
    rows: usize,
    inter: usize,
    visible: usize,
    shared: bool,
    limit: f32,
    gate_bits: Vec<u32>,
    up_bits: Vec<u32>,
    route_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    grad_gate_bits: Vec<u32>,
    grad_up_bits: Vec<u32>,
    grad_route_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Fp32Clamp {
    x_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    w1_bits: Vec<u32>,
    w2_bits: Vec<u32>,
    w3_bits: Vec<u32>,
    output_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
}

fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_mixed_moe_grad.json"
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
fn gate(label: &str, actual: &[f32], expected: &[u32], failures: &mut Vec<String>) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut worst = 0.0f64;
    let mut at = 0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let b = f32::from_bits(b);
        assert!(a.is_finite() && b.is_finite(), "{label} non-finite");
        let drift = (a as f64 - b as f64).abs();
        if drift > worst {
            worst = drift;
            at = i;
        }
    }
    eprintln!(
        "V4.1 mixed MoE adjoint {label}: {} values, worst {worst:.17e}",
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
fn check_gradient(
    c: &Case,
    gradient: &DeepseekV41MixedMoeGradients,
    failures: &mut Vec<String>,
) -> Result<()> {
    gate(
        &format!("{} dX", c.name),
        &gradient.grad_input.read_f32(c.grad_x_bits.len())?,
        &c.grad_x_bits,
        failures,
    );
    gate(
        &format!("{} dRoute", c.name),
        &gradient
            .grad_route_weights
            .read_f32(c.grad_route_bits.len())?,
        &c.grad_route_bits,
        failures,
    );
    let actual = gradient.adapter_gradients("layers.0.ffn")?;
    assert_eq!(
        actual.len(),
        c.gradients.values().filter(|v| v.is_some()).count(),
        "{} active target count",
        c.name
    );
    for g in actual {
        let target = g.target.strip_prefix("layers.0.ffn.").unwrap();
        let expected = c.gradients[target]
            .as_ref()
            .expect("an inactive expert must have None gradients");
        gate(
            &format!("{} {target} dA", c.name),
            &g.grad_a,
            &expected.a_bits,
            failures,
        );
        gate(
            &format!("{} {target} dB", c.name),
            &g.grad_b,
            &expected.b_bits,
            failures,
        );
    }
    Ok(())
}

#[test]
fn mixed_moe_retained_chain_and_adapter_vjps_match_cpu_and_release_experts() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41MixedMoeAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().cases {
        let moe = c.native(&device, true)?;
        let input = gpu(&device, &c.x_bits)?;
        let inference = moe.apply(&input, c.rows)?;
        let tape = adjoints.forward(&moe, &input, c.rows)?;
        assert_eq!(
            inference.read_u32(c.output_bits.len())?,
            tape.output.read_u32(c.output_bits.len())?,
            "{} actual inference identity",
            c.name
        );
        gate(
            &format!("{} forward", c.name),
            &tape.output.read_f32(c.output_bits.len())?,
            &c.output_bits,
            &mut failures,
        );
        assert_eq!(
            tape.indices.read_u32(c.indices.len())?,
            c.indices,
            "{} selected indices",
            c.name
        );
        assert_eq!(
            tape.counts.read_u32(c.counts.len())?,
            c.counts,
            "{} counts",
            c.name
        );
        gate(
            &format!("{} route", c.name),
            &tape.route.read_f32(c.route_bits.len())?,
            &c.route_bits,
            &mut failures,
        );
        input.write_f32(&vec![7.0; c.x_bits.len()])?;
        let before = device.memory_stats()?.live_buffer_bytes;
        drop(moe);
        let after = device.memory_stats()?.live_buffer_bytes;
        let frozen_payload: usize = c
            .experts
            .iter()
            .chain([&c.shared])
            .flat_map(|e| [&e.w1, &e.w2, &e.w3])
            .map(|w| w.data.len() + w.scales.len())
            .sum();
        assert!(
            before - after >= frozen_payload,
            "{} tape pinned frozen experts",
            c.name
        );
        // No adapter in the restaged view: backward must use the retained
        // evaluation's factor snapshots, including shared and unused experts.
        let restaged = c.native(&device, false)?;
        let gradient = adjoints.backward(&restaged, &tape, &gpu(&device, &c.upstream_bits)?)?;
        check_gradient(&c, &gradient, &mut failures)?;
        if c.name == "sqrtsoftplus-block128" {
            for id in 0..c.config.experts {
                let compact = GpuBuffer::zeros_f32(&device, c.rows * c.config.hidden_size)?;
                let weights = GpuBuffer::zeros_f32(&device, c.rows)?;
                let mapping = GpuBuffer::zeros_u32(&device, c.rows)?;
                let push = MoePackPush {
                    rows: c.rows as u32,
                    dim: c.config.hidden_size as u32,
                    topk: c.config.activated as u32,
                    expert: id as u32,
                };
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                restaged.pack.record_dispatch(
                    &mut commands,
                    &[
                        &gpu(&device, &c.upstream_bits)?,
                        &tape.indices,
                        &tape.route,
                        &compact,
                        &weights,
                        &mapping,
                    ],
                    bytemuck::bytes_of(&push),
                    [c.config.hidden_size.div_ceil(64) as u32, c.rows as u32, 1],
                )?;
                commands.submit()?;
                let (_, w2, _, _) = adjoints.with_expert(&restaged, id as u32, |w| {
                    adjoints.expert_backward(
                        &restaged,
                        &tape,
                        tape.experts[id].as_ref().unwrap(),
                        id as u32,
                        w,
                        &compact,
                    )
                })?;
                let expected = &c.trace[&format!("experts.{id}.w2")].grad_input_bits;
                gate(
                    &format!("{} expert {id} down input VJP", c.name),
                    &w2.grad_input.read_f32(expected.len())?,
                    expected,
                    &mut failures,
                );
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn fp32_moe_scalar_clamp_bounds_match_pinned_cpu_subgradient() -> Result<()> {
    let c = fixture().fp32_clamp;
    let device = VulkanDevice::new()?;
    let moe = DeepseekV41Moe::new(
        device.clone(),
        DeepseekV41MoeConfig {
            hidden_size: 4,
            intermediate_size: 4,
            experts: 1,
            activated: 1,
            score_func: DeepseekV41MoeScoreFunc::Softmax,
            gate_temp: 1.0,
            norm_topk_prob: true,
            route_scale: 1.0,
            swiglu_limit: 0.5,
        },
        DeepseekV41MoeWeights {
            gate_weight: vec![0.0; 4],
            gate_bias: vec![0.0],
            expert_w1: vec![values(&c.w1_bits)],
            expert_w2: vec![values(&c.w2_bits)],
            expert_w3: vec![values(&c.w3_bits)],
            shared_w1: vec![0.0; 16],
            shared_w2: vec![0.0; 16],
            shared_w3: vec![0.0; 16],
        },
    )?;
    let input = gpu(&device, &c.x_bits)?;
    let mut failures = Vec::new();
    gate(
        "FP32 scalar clamp forward",
        &moe.apply(&input, 4)?.read_f32(16)?,
        &c.output_bits,
        &mut failures,
    );
    let gradient = DeepseekV41BlockAdjoints::new(device.clone())?.moe(
        &moe,
        &gpu(&device, &c.upstream_bits)?,
        &input,
        4,
    )?;
    gate(
        "FP32 scalar clamp dX",
        &gradient.read_f32(16)?,
        &c.grad_x_bits,
        &mut failures,
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn mixed_swiglu_clamp_route_reduction_and_padding_match_cpu() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41MixedMoeAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().swiglu {
        let dg = GpuBuffer::zeros_f32(&device, c.rows * c.inter)?;
        let du = GpuBuffer::zeros_f32(&device, c.rows * c.inter)?;
        let dr = GpuBuffer::zeros_f32(&device, c.rows)?;
        let push = MixedSwigluPush {
            rows: c.rows as u32,
            inter: c.inter as u32,
            expert: if c.shared { u32::MAX } else { 0 },
            bf16: 1,
            limit: c.limit,
        };
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        adjoints.swiglu.record_dispatch(
            &mut commands,
            &[
                &gpu(&device, &c.gate_bits)?,
                &gpu(&device, &c.up_bits)?,
                &gpu(&device, &c.route_bits)?,
                &GpuBuffer::from_u32(&device, &[c.visible as u32])?,
                &gpu(&device, &c.upstream_bits)?,
                &dg,
                &du,
                &dr,
            ],
            bytemuck::bytes_of(&push),
            [c.rows.div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        for (label, actual, expected) in [
            ("dGate", dg.read_f32(c.rows * c.inter)?, &c.grad_gate_bits),
            ("dUp", du.read_f32(c.rows * c.inter)?, &c.grad_up_bits),
            ("dRoute", dr.read_f32(c.rows)?, &c.grad_route_bits),
        ] {
            gate(
                &format!("{} {label}", c.name),
                &actual,
                expected,
                &mut failures,
            );
        }
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
    fn new(c: &Case) -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "hierarchos-v41-mixed-adjoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let owned = Self(root);
        let mut config: serde_json::Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
        ))?;
        for (key, value) in [
            ("hidden_size", c.config.hidden_size),
            ("moe_intermediate_size", c.config.intermediate_size),
            ("n_routed_experts", c.config.experts),
            ("num_experts_per_tok", c.config.activated),
        ] {
            config["text_config"][key] = serde_json::json!(value);
        }
        fs::write(owned.0.join("config.json"), serde_json::to_vec(&config)?)?;
        let mut tensors = Vec::new();
        for (id, e) in c.experts.iter().enumerate() {
            pack_expert(&mut tensors, &format!("layers.0.ffn.experts.{id}"), e);
        }
        pack_expert(&mut tensors, "layers.0.ffn.shared_experts", &c.shared);
        tensors.push((
            "layers.0.ffn.gate.weight".into(),
            "F32".into(),
            vec![c.config.experts, c.config.hidden_size],
            c.gate_weight.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        tensors.push((
            "layers.0.ffn.gate.bias".into(),
            "F32".into(),
            vec![c.config.experts],
            c.gate_bias.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        write_tensors(&owned.0.join("model.safetensors"), tensors)?;
        let rank = c.shared.w1.rank;
        fs::write(
            owned.0.join("adapter_config.json"),
            serde_json::to_vec(&serde_json::json!({
                "peft_type":"LORA","task_type":"CAUSAL_LM","r":rank,"lora_alpha":rank as f32*0.5,
                "target_modules":["w1","w2","w3"],"lora_dropout":0.0,"bias":"none","inference_mode":false
            }))?,
        )?;
        let mut adapters = Vec::new();
        for (name, e) in c
            .experts
            .iter()
            .enumerate()
            .map(|(i, e)| (format!("experts.{i}"), e))
            .chain([("shared_experts".into(), &c.shared)])
        {
            for (leaf, w) in [("w1", &e.w1), ("w2", &e.w2), ("w3", &e.w3)] {
                for (label, shape, bits) in [
                    ("A", vec![rank, w.cols], &w.a_bits),
                    ("B", vec![w.rows, rank], &w.b_bits),
                ] {
                    adapters.push((
                        format!("base_model.model.layers.0.ffn.{name}.{leaf}.lora_{label}.weight"),
                        "F32".into(),
                        shape,
                        bits.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ));
                }
            }
        }
        write_tensors(&owned.0.join("adapter_model.safetensors"), adapters)?;
        Ok(owned)
    }
    fn streamed(
        &self,
        c: &Case,
        device: &VulkanDevice,
        adapters: bool,
        shared: bool,
    ) -> Result<DeepseekV41MixedMoe> {
        let mut checkpoint = DeepseekV41Checkpoint::open(&self.0)?;
        if adapters {
            checkpoint.set_lora_adapter(Arc::new(DeepseekV41LoraAdapter::load(&self.0)?))?;
        }
        let weights = checkpoint.load_streamed_moe(0, shared)?;
        DeepseekV41MixedMoe::new_streamed(
            device.clone(),
            c.config.native()?,
            weights,
            c.precision(),
        )
    }
}
type Tensor = (String, String, Vec<usize>, Vec<u8>);
fn pack_expert(tensors: &mut Vec<Tensor>, prefix: &str, e: &Expert) {
    for (leaf, w) in [("w1", &e.w1), ("w2", &e.w2), ("w3", &e.w3)] {
        let fp4 = w.dtype == "fp4_row32";
        tensors.push((
            format!("{prefix}.{leaf}.weight"),
            if fp4 { "I8" } else { "F8_E4M3" }.into(),
            vec![w.rows, if fp4 { w.cols / 2 } else { w.cols }],
            w.data.clone(),
        ));
        tensors.push((
            format!("{prefix}.{leaf}.scale"),
            "F8_E8M0".into(),
            w.format().scale_shape(w.rows, w.cols).to_vec(),
            w.scales.clone(),
        ));
    }
}
fn write_tensors(path: &std::path::Path, tensors: Vec<Tensor>) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = payload.len();
        payload.extend_from_slice(&bytes);
        header.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&header)?;
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend(json);
    file.extend(payload);
    fs::write(path, file)?;
    Ok(())
}

#[test]
fn mixed_moe_checkpoint_streaming_backward_and_inactive_gradients_match_cpu() -> Result<()> {
    let c = fixture().cases.remove(0);
    assert!(c.counts.contains(&0));
    let package = Package::new(&c)?;
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41MixedMoeAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for shared in [false, true] {
        let moe = package.streamed(&c, &device, true, shared)?;
        assert_eq!(moe.resident_routed_expert_count(), 0);
        assert_eq!(moe.resident_shared_expert_count(), usize::from(!shared));
        let input = gpu(&device, &c.x_bits)?;
        let tape = adjoints.forward(&moe, &input, c.rows)?;
        assert_eq!(
            tape.output.read_u32(c.output_bits.len())?,
            moe.apply(&input, c.rows)?.read_u32(c.output_bits.len())?,
            "streamed retained forward must match actual inference"
        );
        gate(
            &format!("{} streamed shared={shared} forward", c.name),
            &tape.output.read_f32(c.output_bits.len())?,
            &c.output_bits,
            &mut failures,
        );
        drop(moe);
        let restaged = package.streamed(&c, &device, false, shared)?;
        let gradient = adjoints.backward(&restaged, &tape, &gpu(&device, &c.upstream_bits)?)?;
        check_gradient(&c, &gradient, &mut failures)?;
        assert!(gradient.adapter_gradients("").is_err());
        if shared {
            let source = package.0.join("model.safetensors");
            let parked = package.0.join("model.safetensors.unavailable");
            fs::rename(&source, &parked)?;
            let failed = adjoints
                .backward(&restaged, &tape, &gpu(&device, &c.upstream_bits)?)
                .is_err();
            fs::rename(&parked, &source)?;
            assert!(
                failed,
                "streamed backward must require readable frozen payloads"
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn mixed_moe_adjoint_rejects_bad_geometry_config_and_foreign_buffers() -> Result<()> {
    let c = fixture().cases.remove(0);
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41MixedMoeAdjoints::new(device.clone())?;
    let moe = c.native(&device, true)?;
    let input = gpu(&device, &c.x_bits)?;
    assert!(adjoints.forward(&moe, &input, 0).is_err());
    assert!(adjoints.forward(&moe, &input, 65_536).is_err());
    let too_small = GpuBuffer::zeros_f32(&device, 1)?;
    assert!(adjoints.forward(&moe, &too_small, c.rows).is_err());
    let tape = adjoints.forward(&moe, &input, c.rows)?;
    assert!(adjoints.backward(&moe, &tape, &too_small).is_err());
    let mut other = c.native(&device, false)?;
    other.config.route_scale += 0.5;
    assert!(adjoints.backward(&other, &tape, &input).is_err());
    // Reject routed storage overflow before allocation even when token-stream
    // and slot lengths independently fit u32 addressing.
    other.config.hidden_size = 2_000_000;
    other.config.activated = 2;
    assert!(format!("{:#}", adjoints.geometry(&other, 2_048).err().unwrap()).contains("routed"));
    let foreign = VulkanDevice::new()?;
    assert!(adjoints
        .forward(&moe, &gpu(&foreign, &c.x_bits)?, c.rows)
        .is_err());
    assert!(adjoints
        .backward(&moe, &tape, &gpu(&foreign, &c.upstream_bits)?)
        .is_err());
    Ok(())
}

#[test]
fn mixed_moe_computed_conditional_adamw_resume_and_export_match_cpu() -> Result<()> {
    use crate::deepseek_v41::train::conditional_tests as qualification;
    use crate::deepseek_v41::DeepseekV41AdapterTrainer;
    use sha2::{Digest, Sha256};
    let oracle = qualification::mixed_fixture();
    assert_eq!(
        oracle.parent_sha256,
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../validation/fixtures/deepseek_v41_mixed_moe_grad.json"
            ))
        )
    );
    let c = fixture().cases.remove(oracle.case_index);
    assert_eq!(c.name, oracle.name);
    let package = Package::new(&c)?;
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41MixedMoeAdjoints::new(device.clone())?;
    let initial = DeepseekV41LoraAdapter::load(&package.0)?;
    let mut trainer = DeepseekV41AdapterTrainer::new(&initial, oracle.hyper)?;
    let mut resumed = None;
    let model = |trainer: &DeepseekV41AdapterTrainer| -> Result<DeepseekV41MixedMoe> {
        let mut checkpoint = DeepseekV41Checkpoint::open(&package.0)?;
        checkpoint.set_lora_adapter(Arc::new(trainer.adapter()?))?;
        DeepseekV41MixedMoe::new_streamed(
            device.clone(),
            c.config.native()?,
            checkpoint.load_streamed_moe(0, true)?,
            c.precision(),
        )
    };
    let run = |trainer: &mut DeepseekV41AdapterTrainer, index: usize| -> Result<Vec<u32>> {
        let expected = &oracle.steps[index];
        let active = model(trainer)?;
        assert_eq!(active.resident_routed_expert_count(), 0);
        assert_eq!(active.resident_shared_expert_count(), 0);
        let tape = adjoints.forward(&active, &gpu(&device, &c.x_bits)?, c.rows)?;
        drop(active);
        qualification::gate(
            &format!("mixed step {index} output"),
            &tape.output.read_f32(expected.output_bits.len())?,
            &expected.output_bits,
        );
        // Frozen weights and old adapter uploads are gone. Backward uses the
        // retained factor snapshots and checkpoint-stages base payloads only.
        let frozen = package.streamed(&c, &device, false, true)?;
        let gradient = adjoints.backward(&frozen, &tape, &gpu(&device, &c.upstream_bits)?)?;
        drop(frozen);
        let gradients = gradient.adapter_gradients("layers.0.ffn")?;
        assert_eq!(gradients.len(), expected.gradients.len());
        assert!(gradients.len() < trainer.targets().count());
        let before = trainer.clone();
        for g in &gradients {
            let e = &expected.gradients[&g.target];
            qualification::gate(
                &format!("mixed step {index} {} dA", g.target),
                &g.grad_a,
                e.a_bits.as_ref().unwrap(),
            );
            qualification::gate(
                &format!("mixed step {index} {} dB", g.target),
                &g.grad_b,
                e.b_bits.as_ref().unwrap(),
            );
        }
        trainer.step(&gradients)?;
        for target in before
            .targets()
            .filter(|target| !expected.gradients.contains_key(*target))
        {
            assert_eq!(
                before.parameter_step_counts(target),
                trainer.parameter_step_counts(target)
            );
            assert_eq!(before.factors(target), trainer.factors(target));
            assert_eq!(before.moments(target), trainer.moments(target));
        }
        qualification::compare_state(&format!("mixed step {index}"), trainer, &expected.states);
        let output = model(trainer)?.apply(&gpu(&device, &c.x_bits)?, c.rows)?;
        qualification::gate(
            &format!("mixed step {index} updated output"),
            &output.read_f32(expected.updated_output_bits.len())?,
            &expected.updated_output_bits,
        );
        output.read_u32(expected.updated_output_bits.len())
    };
    for index in 0..oracle.steps.len() {
        let actual = run(&mut trainer, index)?;
        if index == 1 {
            let checkpoint = package.0.join("conditional-training-checkpoint");
            trainer.save_checkpoint(&checkpoint)?;
            resumed = Some(DeepseekV41AdapterTrainer::resume_from_checkpoint(
                &checkpoint,
            )?);
        }
        if index == 2 {
            let restored = resumed.as_mut().unwrap();
            let replay = run(restored, index)?;
            qualification::assert_exact(&trainer, restored);
            assert_eq!(actual, replay, "resumed updated output bits");
        }
    }
    let export = package.0.join("trained-export");
    trainer.adapter()?.export(&export)?;
    let exported =
        DeepseekV41AdapterTrainer::new(&DeepseekV41LoraAdapter::load(&export)?, oracle.hyper)?;
    assert_eq!(
        model(&trainer)?
            .apply(&gpu(&device, &c.x_bits)?, c.rows)?
            .read_u32(c.output_bits.len())?,
        model(&exported)?
            .apply(&gpu(&device, &c.x_bits)?, c.rows)?
            .read_u32(c.output_bits.len())?
    );
    Ok(())
}
