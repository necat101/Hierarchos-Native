use super::*;
use crate::deepseek_v41::indexer::LinearKernel;
use crate::deepseek_v41::peft::DeepseekV41LoraHost;
use crate::deepseek_v41::{
    DeepseekV41AttentionProjectionAdjoints, DeepseekV41AttentionTape,
    DeepseekV41AttentionWindowView, DeepseekV41IndexSelection, DeepseekV41PackedWeights,
    DeepseekV41WeightFormat,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    tolerance: f64,
    rtol: f64,
    comparison: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    storage: String,
    batch: usize,
    hidden: usize,
    qr: usize,
    heads: usize,
    dim: usize,
    groups: usize,
    output_rank: usize,
    window: usize,
    ratio: usize,
    capacity: usize,
    q_norm_bits: Vec<u32>,
    kv_norm_bits: Vec<u32>,
    wo_a_bits: Vec<u32>,
    sink_bits: Vec<u32>,
    head_bits: Vec<u32>,
    compressed_bits: Vec<u32>,
    grad_compressed_bits: Option<Vec<u32>>,
    grad_initial_window_bits: Option<Vec<u32>>,
    projections: BTreeMap<String, Projection>,
    steps: Vec<Step>,
    gradients: BTreeMap<String, Pair>,
}
#[derive(Deserialize)]
struct Projection {
    weight: Weight,
    rank: usize,
    scale: f32,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Weight {
    rows: usize,
    cols: usize,
    dtype: String,
    #[serde(default)]
    values_bits: Vec<u32>,
    #[serde(default)]
    data: Vec<u8>,
    #[serde(default)]
    scales: Vec<u8>,
}
#[derive(Deserialize)]
struct Pair {
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Step {
    position: usize,
    seq: usize,
    x_bits: Vec<u32>,
    query_bits: Vec<u32>,
    latent_bits: Vec<u32>,
    window_kv_bits: Vec<u32>,
    sparse_bits: Vec<u32>,
    output_bits: Vec<u32>,
    logits_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    wi: Vec<i32>,
    ci: Vec<i32>,
    topk: usize,
    compressed_topk: usize,
    compressed_width: usize,
    seed: bool,
    extra_latent_bits: Option<Vec<u32>>,
    grad_x_bits: Option<Vec<u32>>,
    grad_latent_bits: Option<Vec<u32>>,
    grad_query_bits: Option<Vec<u32>>,
    grad_window_kv_bits: Option<Vec<u32>>,
    grad_sparse_bits: Option<Vec<u32>>,
}
fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_attention_projection_grad.json"
    ))
    .unwrap();
    assert_eq!(
        (f.tolerance, f.rtol, f.comparison.as_str()),
        (2e-7, 0.0, "strict_less_than")
    );
    f
}
fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}
fn gpu(device: &VulkanDevice, bits: &[u32]) -> Result<GpuBuffer> {
    if bits.is_empty() {
        GpuBuffer::zeros_u32(device, 1)
    } else {
        GpuBuffer::from_u32(device, bits)
    }
}
fn indices(device: &VulkanDevice, values: &[i32]) -> Result<GpuBuffer> {
    gpu(
        device,
        &values.iter().map(|&n| n as u32).collect::<Vec<_>>(),
    )
}
impl Case {
    fn native(&self, device: &VulkanDevice, adapters: bool) -> Result<DeepseekV41Attention> {
        let projection = |name: &str| -> Result<DeepseekV41LinearWeights> {
            let p = &self.projections[name];
            let w = &p.weight;
            let lora = adapters.then(|| DeepseekV41LoraHost {
                rank: p.rank,
                scale: p.scale,
                a: values(&p.a_bits).into(),
                b: values(&p.b_bits).into(),
            });
            Ok(match w.dtype.as_str() {
                "fp32" => DeepseekV41LinearWeights::Fp32 {
                    rows: w.rows,
                    cols: w.cols,
                    values: values(&w.values_bits),
                    lora,
                },
                "bf16" => DeepseekV41LinearWeights::Bf16 {
                    rows: w.rows,
                    cols: w.cols,
                    values: values(&w.values_bits),
                    lora,
                },
                "fp8_block32" => DeepseekV41LinearWeights::Packed {
                    weight: DeepseekV41PackedWeights::new(
                        w.rows,
                        w.cols,
                        DeepseekV41WeightFormat::Fp8Block32,
                        w.data.clone(),
                        w.scales.clone(),
                    )?,
                    lora,
                },
                _ => unreachable!(),
            })
        };
        let wo_a = if self.storage == "fp32" {
            DeepseekV41LinearWeights::Fp32 {
                rows: self.groups * self.output_rank,
                cols: self.heads * self.dim / self.groups,
                values: values(&self.wo_a_bits),
                lora: None,
            }
        } else {
            DeepseekV41LinearWeights::Bf16 {
                rows: self.groups * self.output_rank,
                cols: self.heads * self.dim / self.groups,
                values: values(&self.wo_a_bits),
                lora: None,
            }
        };
        DeepseekV41Attention::new_mixed(
            device.clone(),
            DeepseekV41AttentionConfig {
                heads: self.heads,
                head_dim: self.dim,
                hidden_size: self.hidden,
                q_lora_rank: self.qr,
                o_groups: self.groups,
                o_lora_rank: self.output_rank,
                window_size: self.window,
                compress_ratio: self.ratio,
                norm_eps: 1e-6,
                rotary: DeepseekV41RotaryConfig::sliding(16, 64, 10000.0),
            },
            DeepseekV41MixedAttentionWeights {
                wq_a: projection("wq_a")?,
                q_norm: values(&self.q_norm_bits),
                wq_b: projection("wq_b")?,
                wkv: projection("wkv")?,
                kv_norm: values(&self.kv_norm_bits),
                wo_a,
                wo_b: projection("wo_b")?,
                attn_sink: values(&self.sink_bits),
            },
            self.batch,
            if self.storage == "fp32" {
                DeepseekV41WeightOutput::Fp32
            } else {
                DeepseekV41WeightOutput::Bf16
            },
        )
    }
}
fn gate(label: &str, actual: &[f32], expected: &[u32], failures: &mut Vec<String>) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut worst = 0.0f64;
    let mut at = 0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let b = f32::from_bits(b);
        assert!(a.is_finite() && b.is_finite(), "{label} nonfinite at {i}");
        let d = (a as f64 - b as f64).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!(
        "V4.1 attention projection {label}: {} values, worst {worst:.17e} at {at}",
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
fn buffer_gate(
    label: &str,
    actual: &GpuBuffer,
    expected: &[u32],
    failures: &mut Vec<String>,
) -> Result<()> {
    gate(label, &actual.read_f32(expected.len())?, expected, failures);
    Ok(())
}
fn optional_gate(
    label: &str,
    actual: Option<&GpuBuffer>,
    expected: &Option<Vec<u32>>,
    failures: &mut Vec<String>,
) -> Result<()> {
    match (actual, expected) {
        (Some(a), Some(b)) => buffer_gate(label, a, b, failures),
        (None, None) => Ok(()),
        _ => panic!("{label} None mismatch"),
    }
}
fn accumulate(
    targets: &mut BTreeMap<String, DeepseekV41AdapterGradient>,
    gradients: Vec<DeepseekV41AdapterGradient>,
) {
    for g in gradients {
        if let Some(t) = targets.get_mut(&g.target) {
            assert_eq!(
                (t.grad_a.len(), t.grad_b.len()),
                (g.grad_a.len(), g.grad_b.len())
            );
            for (a, b) in t.grad_a.iter_mut().zip(g.grad_a) {
                *a += b;
            }
            for (a, b) in t.grad_b.iter_mut().zip(g.grad_b) {
                *a += b;
            }
        } else {
            targets.insert(g.target.clone(), g);
        }
    }
}

#[test]
fn attention_projection_cache_chain_and_all_factor_vjps_match_cpu() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41AttentionProjectionAdjoints::new(device.clone())?;
    let head_linear = LinearKernel::new(&device)?;
    let mut failures = Vec::new();
    for c in fixture().cases {
        let mut attention = c.native(&device, true)?;
        let mut inference = c.native(&device, true)?;
        let window = DeepseekV41WindowKvCache::new(device.clone(), c.batch, c.window, c.dim)?;
        let compressed = if c.capacity > 0 {
            let cache =
                DeepseekV41CompressedKvCache::new(device.clone(), c.batch, c.capacity, c.dim)?;
            cache.values().write_u32(&c.compressed_bits)?;
            Some(cache)
        } else {
            None
        };
        let mut cache_tape = DeepseekV41AttentionTape::new(device.clone());
        let window_id = cache_tape.register_window(&window)?;
        let compressed_id = compressed
            .as_ref()
            .map(|cache| cache_tape.register_compressed(cache))
            .transpose()?;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let head = gpu(&device, &c.head_bits)?;
        for (i, s) in c.steps.iter().enumerate() {
            let label = format!("{} step {i}", c.name);
            let x = gpu(&device, &s.x_bits)?;
            let input = adjoints.forward_inputs(&attention, &x, c.batch, s.seq, s.position)?;
            buffer_gate(
                &format!("{label} query"),
                &input.query,
                &s.query_bits,
                &mut failures,
            )?;
            buffer_gate(
                &format!("{label} latent"),
                &input.query_latent,
                &s.latent_bits,
                &mut failures,
            )?;
            buffer_gate(
                &format!("{label} window"),
                &input.window_kv,
                &s.window_kv_bits,
                &mut failures,
            )?;
            let (write, view) = if i == 0 {
                let write = cache_tape.seed_window(&window_id, &input.window_kv, s.seq)?;
                let view = DeepseekV41AttentionWindowView::Prefill(write.clone());
                (write, view)
            } else {
                let write = cache_tape.write_window(&window_id, &input.window_kv, s.position)?;
                (
                    write,
                    DeepseekV41AttentionWindowView::Cache(window_id.clone()),
                )
            };
            let wi = indices(&device, &s.wi)?;
            let ci = indices(&device, &s.ci)?;
            let read = cache_tape.attend(
                &attention,
                &input.query,
                view,
                compressed_id.as_ref(),
                &wi,
                &ci,
                DeepseekV41SparseAttentionParams {
                    batch: c.batch,
                    seq: s.seq,
                    window_rows: if i == 0 { s.seq } else { c.window },
                    window_topk: s.topk,
                    compressed_topk: s.compressed_topk,
                    compressed_capacity: c.capacity,
                },
            )?;
            buffer_gate(
                &format!("{label} sparse"),
                &read.output,
                &s.sparse_bits,
                &mut failures,
            )?;
            let output =
                adjoints.forward_output(&attention, &read.output, c.batch, s.seq, s.position)?;
            buffer_gate(
                &format!("{label} output"),
                &output.output,
                &s.output_bits,
                &mut failures,
            )?;
            let logits = head_linear.apply(
                &device,
                &output.output,
                &head,
                c.batch * s.seq,
                c.hidden,
                11,
            )?;
            buffer_gate(
                &format!("{label} logits"),
                &logits,
                &s.logits_bits,
                &mut failures,
            )?;
            let selection = (c.ratio > 0)
                .then(|| -> Result<DeepseekV41IndexSelection> {
                    Ok(DeepseekV41IndexSelection {
                        rows: c.batch * s.seq,
                        width: s.compressed_width,
                        topk: s.compressed_topk,
                        scores: GpuBuffer::zeros_f32(
                            &device,
                            (c.batch * s.seq * s.compressed_width).max(1),
                        )?,
                        indices: Some(ci.clone()),
                        candidates: None,
                    })
                })
                .transpose()?;
            let normal = if i == 0 {
                inference.prefill(&x, compressed.as_ref(), selection.as_ref(), c.batch, s.seq)?
            } else {
                inference.decode_step(
                    &x,
                    compressed.as_ref(),
                    selection.as_ref(),
                    c.batch,
                    s.position,
                )?
            };
            assert_eq!(
                output.output.read_u32(s.output_bits.len())?,
                normal.read_u32(s.output_bits.len())?,
                "{label} native inference identity"
            );
            x.write_f32(&vec![17.0; s.x_bits.len()])?;
            // Public forward products are not the private norm/projection tape.
            input
                .query_latent
                .write_f32(&vec![17.0; s.latent_bits.len()])?;
            input
                .window_kv
                .write_f32(&vec![17.0; s.window_kv_bits.len()])?;
            input.query.write_f32(&vec![17.0; s.query_bits.len()])?;
            outputs.push(output);
            inputs.push(input);
            reads.push(read.id);
            writes.push(write);
        }
        // No frozen base is retained. All projections can be retired and
        // restaged without adapters; backward uses the recorded factor copies.
        drop(inference);
        let before = device.memory_stats()?.live_buffer_bytes;
        let projections = attention.projections.take().unwrap();
        projections.q_norm.write_f32(&vec![17.0; c.qr])?;
        projections.kv_norm.write_f32(&vec![17.0; c.dim])?;
        drop(projections);
        let after = device.memory_stats()?.live_buffer_bytes;
        let frozen_bytes: usize = c
            .projections
            .values()
            .map(|p| {
                if p.weight.dtype == "fp8_block32" {
                    p.weight.data.len()
                } else {
                    p.weight.rows * p.weight.cols * 4
                }
            })
            .sum::<usize>()
            + c.wo_a_bits.len() * 4;
        assert!(
            before - after >= frozen_bytes,
            "{} tape pinned frozen bases",
            c.name
        );
        eprintln!("V4.1 attention projection {} retired projection bytes {}, frozen minimum {frozen_bytes}",c.name,before-after);
        drop(attention);
        let reloaded = c.native(&device, false)?;
        let mut factor_gradients = BTreeMap::new();
        let mut seeds = Vec::new();
        for (i, s) in c.steps.iter().enumerate().rev() {
            if s.seed {
                let grad = adjoints.backward_output(
                    &reloaded,
                    &outputs[i],
                    &gpu(&device, &s.upstream_bits)?,
                )?;
                optional_gate(
                    &format!("{} step {i} dSparse", c.name),
                    grad.grad_input.as_ref(),
                    &s.grad_sparse_bits,
                    &mut failures,
                )?;
                accumulate(
                    &mut factor_gradients,
                    grad.adapter_gradients("layers.0.attn")?,
                );
                seeds.push((reads[i].clone(), grad.grad_input.unwrap()));
            }
        }
        let cache_grad = cache_tape.backward(&seeds)?;
        for (i, s) in c.steps.iter().enumerate().rev() {
            let query = cache_grad.query(&reads[i])?;
            let source = cache_grad.source(&writes[i])?;
            optional_gate(
                &format!("{} step {i} dQuery", c.name),
                query,
                &s.grad_query_bits,
                &mut failures,
            )?;
            optional_gate(
                &format!("{} step {i} dWindow", c.name),
                source,
                &s.grad_window_kv_bits,
                &mut failures,
            )?;
            let extra = s
                .extra_latent_bits
                .as_ref()
                .map(|v| gpu(&device, v))
                .transpose()?;
            let grad =
                adjoints.backward_inputs(&reloaded, &inputs[i], query, source, extra.as_ref())?;
            optional_gate(
                &format!("{} step {i} dX", c.name),
                grad.grad_input.as_ref(),
                &s.grad_x_bits,
                &mut failures,
            )?;
            optional_gate(
                &format!("{} step {i} dLatent", c.name),
                grad.grad_query_latent.as_ref(),
                &s.grad_latent_bits,
                &mut failures,
            )?;
            accumulate(
                &mut factor_gradients,
                grad.adapter_gradients("layers.0.attn")?,
            );
        }
        optional_gate(
            &format!("{} dInitialWindow", c.name),
            cache_grad.initial_cache(&window_id)?,
            &c.grad_initial_window_bits,
            &mut failures,
        )?;
        if let Some(id) = compressed_id {
            optional_gate(
                &format!("{} dCompressed", c.name),
                cache_grad.initial_cache(&id)?,
                &c.grad_compressed_bits,
                &mut failures,
            )?;
        }
        assert_eq!(factor_gradients.len(), 4);
        for (target, grad) in factor_gradients {
            let module = target.strip_prefix("layers.0.attn.").unwrap();
            let expected = &c.gradients[module];
            gate(
                &format!("{} {module} dA", c.name),
                &grad.grad_a,
                &expected.a_bits,
                &mut failures,
            );
            gate(
                &format!("{} {module} dB", c.name),
                &grad.grad_b,
                &expected.b_bits,
                &mut failures,
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn attention_projection_conditional_branches_preserve_none_gradients() -> Result<()> {
    let device = VulkanDevice::new()?;
    let c = fixture().cases.remove(0);
    let a = c.native(&device, true)?;
    let adjoints = DeepseekV41AttentionProjectionAdjoints::new(device.clone())?;
    let s = &c.steps[0];
    let t = adjoints.forward_inputs(&a, &gpu(&device, &s.x_bits)?, c.batch, s.seq, 0)?;
    let q = GpuBuffer::zeros_f32(&device, s.query_bits.len())?;
    let kv = GpuBuffer::zeros_f32(&device, s.window_kv_bits.len())?;
    let latent = GpuBuffer::zeros_f32(&device, s.latent_bits.len())?;
    for (qg, kg, extra, names) in [
        (None, None, None, vec![]),
        (Some(&q), None, None, vec!["wq_a", "wq_b"]),
        (None, Some(&kv), None, vec!["wkv"]),
        (None, None, Some(&latent), vec!["wq_a"]),
    ] {
        let grad = adjoints.backward_inputs(&a, &t, qg, kg, extra)?;
        let adapters = grad.adapter_gradients("layers.0.attn")?;
        assert_eq!(
            adapters
                .iter()
                .map(|g| g.target.strip_prefix("layers.0.attn.").unwrap())
                .collect::<Vec<_>>(),
            names
        );
        if names.is_empty() {
            assert!(grad.grad_input.is_none() && grad.grad_query_latent.is_none());
        } else {
            assert_eq!(
                grad.grad_input.unwrap().read_f32(s.x_bits.len())?,
                vec![0.0; s.x_bits.len()]
            );
        }
        for g in adapters {
            assert!(g.grad_a.iter().chain(&g.grad_b).all(|&v| v == 0.0));
        }
    }
    Ok(())
}

#[test]
fn attention_projection_rejects_geometry_and_configuration_mismatch() -> Result<()> {
    let device = VulkanDevice::new()?;
    let c = fixture().cases.remove(0);
    let mut a = c.native(&device, true)?;
    let adjoints = DeepseekV41AttentionProjectionAdjoints::new(device.clone())?;
    let s = &c.steps[0];
    let x = gpu(&device, &s.x_bits)?;
    let small = GpuBuffer::zeros_f32(&device, 1)?;
    assert!(adjoints.forward_inputs(&a, &x, 0, s.seq, 0).is_err());
    assert!(adjoints
        .forward_inputs(&a, &x, c.batch, s.seq, usize::MAX)
        .is_err());
    assert!(adjoints
        .forward_inputs(&a, &small, c.batch, s.seq, 0)
        .is_err());
    let t = adjoints.forward_inputs(&a, &x, c.batch, s.seq, 0)?;
    assert!(adjoints
        .backward_inputs(&a, &t, Some(&small), None, None)
        .is_err());
    assert!(adjoints
        .backward_inputs(&a, &t, None, Some(&small), None)
        .is_err());
    assert!(adjoints
        .backward_inputs(&a, &t, None, None, Some(&small))
        .is_err());
    assert!(adjoints
        .forward_output(&a, &small, c.batch, s.seq, 0)
        .is_err());
    let output = adjoints.forward_output(&a, &gpu(&device, &s.sparse_bits)?, c.batch, s.seq, 0)?;
    assert!(adjoints.backward_output(&a, &output, &small).is_err());
    a.config.norm_eps = 1e-5;
    assert!(adjoints.backward_inputs(&a, &t, None, None, None).is_err());
    assert!(adjoints.backward_output(&a, &output, &x).is_err());
    assert!(checked(&[usize::MAX, 2]).is_err());
    assert!(checked(&[u32::MAX as usize, 2]).is_err());
    assert!(checked(&[0, 1]).is_err());
    Ok(())
}
