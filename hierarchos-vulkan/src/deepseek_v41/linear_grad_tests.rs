use super::*;

#[test]
fn projection_weight_gradient_cpu_output_geometry_reduction() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        x_bits: Vec<u32>,
        grad_bits: Vec<u32>,
        weight_grad_bits: Vec<u32>,
    }
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_weight_grad_order.json"
    ))?;
    let device = VulkanDevice::new()?;
    let k = DeepseekV41LinearAdjoints::new(device.clone())?;
    if crate::transformer::aten_vector_width() < 16 {
        return Ok(());
    }
    let mut exact = 0usize;
    let mut count = 0usize;
    let mut worst = 0.0f64;
    for c in fixture.cases {
        let x = GpuBuffer::from_f32(
            &device,
            &c.x_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>(),
        )?;
        let g = GpuBuffer::from_f32(
            &device,
            &c.grad_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>(),
        )?;
        let output = GpuBuffer::zeros_f32(&device, c.input_dim * c.output_dim)?;
        let mut batch = vulkan::ComputeBatch::new(&device)?;
        let push = LinearPush {
            rows: c.rows as u32,
            input_dim: c.input_dim as u32,
            output_dim: c.output_dim as u32,
        };
        k.weight_kernel(c.input_dim, c.output_dim).record_dispatch(
            &mut batch,
            &[&x, &g, &output],
            bytemuck::bytes_of(&push),
            weight_grid(c.input_dim, c.output_dim),
        )?;
        batch.submit()?;
        let actual = output.read_u32(c.weight_grad_bits.len())?;
        for (&a, &b) in actual.iter().zip(&c.weight_grad_bits) {
            let drift = (f32::from_bits(a) as f64 - f32::from_bits(b) as f64).abs();
            assert!(
                drift.is_finite() && drift < 2e-7,
                "dW reduction rows={} input={} output={} drift={drift}",
                c.rows,
                c.input_dim,
                c.output_dim
            );
            worst = worst.max(drift);
            count += 1;
            exact += usize::from(a == b);
        }
        // Aligned GEMMs are bit-exact. MKL's odd 3x6 tail has a separate
        // reduction; it is measured under the same strict absolute gate.
        if c.input_dim.is_multiple_of(4) || c.output_dim.is_multiple_of(2) {
            assert_eq!(
                actual, c.weight_grad_bits,
                "aligned dW reduction rows={} input={} output={}",
                c.rows, c.input_dim, c.output_dim
            );
        }
    }
    eprintln!("V4.1 dW geometry: {exact}/{count} bit-exact, worst {worst:.17e}");
    Ok(())
}
use crate::deepseek_v41::indexer::{LinearKernel, NormKernel};
use crate::deepseek_v41::peft::DeepseekV41LoraHost;
use crate::deepseek_v41::{
    DeepseekV41AdamWConfig, DeepseekV41AdapterTrainer, DeepseekV41LinearWeights,
    DeepseekV41LoraAdapter, DeepseekV41PackedWeights, DeepseekV41ResidentLinear,
    DeepseekV41TerminalAdjoints, DeepseekV41WeightFormat, DeepseekV41WeightOutput,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Fixture {
    tolerance: f64,
    comparison: String,
    rtol: f64,
    projection: Vec<Projection>,
    training: Vec<Training>,
}
#[derive(Deserialize)]
struct Projection {
    name: String,
    rows: usize,
    width: usize,
    out: usize,
    rank: usize,
    scale: f32,
    x_bits: Vec<u32>,
    weight_bits: Vec<u32>,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    output_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
    grad_a_bits: Vec<u32>,
    grad_b_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Pair {
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Module {
    target: String,
    rank: usize,
    width: usize,
    out: usize,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct State {
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
    exp_avg_a_bits: Vec<u32>,
    exp_avg_sq_a_bits: Vec<u32>,
    exp_avg_b_bits: Vec<u32>,
    exp_avg_sq_b_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Step {
    logits_bits: Vec<u32>,
    logits_after_bits: Vec<u32>,
    loss: f32,
    grad_x_bits: Vec<u32>,
    gradients: BTreeMap<String, Pair>,
    states: BTreeMap<String, State>,
}
#[derive(Deserialize)]
struct Training {
    name: String,
    rows: usize,
    width: usize,
    latent: usize,
    out: usize,
    vocab: usize,
    x_bits: Vec<u32>,
    w1_bits: Vec<u32>,
    w2_bits: Vec<u32>,
    norm_bits: Vec<u32>,
    head_bits: Vec<u32>,
    targets: Vec<u32>,
    modules: Vec<Module>,
    hyper: DeepseekV41AdamWConfig,
    steps: Vec<Step>,
    adapter_config: serde_json::Value,
}
fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_projection_grad.json"
    ))
    .unwrap();
    assert_eq!(f.tolerance, 2e-7);
    assert_eq!(f.comparison, "strict_less_than");
    assert_eq!(f.rtol, 0.0);
    f
}
fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}
fn assert_bits(left: &[f32], right: &[f32], label: &str) {
    assert_eq!(left.len(), right.len(), "{label}");
    for (i, (left, right)) in left.iter().zip(right).enumerate() {
        assert_eq!(left.to_bits(), right.to_bits(), "{label} at {i}");
    }
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
        "V4.1 projection adjoint {label}: {} values, worst {worst:.17e}",
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
fn base(
    device: &VulkanDevice,
    rows: usize,
    cols: usize,
    bits: &[u32],
    lora: Option<DeepseekV41LoraHost>,
) -> Result<DeepseekV41ResidentLinear> {
    DeepseekV41LinearWeights::Fp32 {
        rows,
        cols,
        values: values(bits),
        lora,
    }
    .upload(device)
}

#[test]
fn projection_tape_lora_vjps_match_cpu_and_release_frozen_weights() -> Result<()> {
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41LinearAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().projection {
        let lora = (c.rank != 0).then(|| DeepseekV41LoraHost {
            rank: c.rank,
            scale: c.scale,
            a: values(&c.a_bits).into(),
            b: values(&c.b_bits).into(),
        });
        let weight = base(&device, c.out, c.width, &c.weight_bits, lora)?;
        let input = gpu(&device, &c.x_bits)?;
        let tape = kernels.forward(&input, c.rows, &weight)?;
        gate(
            &format!("{} forward", c.name),
            &tape.output.read_f32(c.rows * c.out)?,
            &c.output_bits,
            &mut failures,
        );
        // Mutating caller buffers/factors cannot alter this retained evaluation.
        input.write_f32(&vec![7.0; c.rows * c.width])?;
        if let Some(lora) = &weight.lora {
            lora.a.write_f32(&vec![3.0; c.rank * c.width])?;
            lora.b.write_f32(&vec![-3.0; c.out * c.rank])?;
        }
        let before = device.memory_stats()?.live_buffer_bytes;
        drop(weight);
        let after = device.memory_stats()?.live_buffer_bytes;
        assert!(
            before - after >= c.out * c.width * 4,
            "the tape pinned a frozen weight"
        );
        // The reloaded base deliberately has no adapter; backward must use the
        // factor snapshots that actually produced this tape's output.
        let reloaded = base(&device, c.out, c.width, &c.weight_bits, None)?;
        let gradient = kernels.backward(&tape, &reloaded, &gpu(&device, &c.upstream_bits)?)?;
        gate(
            &format!("{} dX", c.name),
            &gradient.grad_input.read_f32(c.rows * c.width)?,
            &c.grad_x_bits,
            &mut failures,
        );
        if c.rank != 0 {
            let grad = gradient.read_adapter_gradient("layers.0.attn.wq_b")?;
            gate(
                &format!("{} dA", c.name),
                &grad.grad_a,
                &c.grad_a_bits,
                &mut failures,
            );
            gate(
                &format!("{} dB", c.name),
                &grad.grad_b,
                &c.grad_b_bits,
                &mut failures,
            );
            if c.name == "zero-B" {
                assert!(grad.grad_a.iter().all(|&v| v == 0.0));
                assert!(grad.grad_b.iter().any(|&v| v != 0.0));
            }
        } else {
            assert!(gradient
                .read_adapter_gradient("layers.0.attn.wq_b")
                .is_err());
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[derive(Deserialize)]
struct PackedFixture {
    tolerance: f64,
    comparison: String,
    rtol: f64,
    cases: Vec<PackedCase>,
}
#[derive(Deserialize)]
struct PackedWeight {
    rows: usize,
    cols: usize,
    dtype: String,
    #[serde(default)]
    data: Vec<u8>,
    #[serde(default)]
    scales: Vec<u8>,
    #[serde(default)]
    values_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct PackedCase {
    name: String,
    rows: usize,
    inner: usize,
    out: usize,
    stream: String,
    rank: usize,
    scale: f32,
    weight: PackedWeight,
    x_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    output_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
    grad_a_bits: Vec<u32>,
    grad_b_bits: Vec<u32>,
}
fn packed_fixture() -> PackedFixture {
    let f: PackedFixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_packed_grad.json"
    ))
    .unwrap();
    assert_eq!(f.tolerance, 2e-7);
    assert_eq!(f.comparison, "strict_less_than");
    assert_eq!(f.rtol, 0.0);
    f
}
fn packed_base(
    device: &VulkanDevice,
    w: &PackedWeight,
    lora: Option<DeepseekV41LoraHost>,
) -> Result<DeepseekV41ResidentLinear> {
    // BF16 dense storage: BF16-exact FP32 values, ordinary FP32 GEMM, one
    // BF16 output rounding, and the same PEFT merge boundary.
    if w.dtype == "bf16_dense" {
        return DeepseekV41LinearWeights::Bf16 {
            rows: w.rows,
            cols: w.cols,
            values: values(&w.values_bits),
            lora,
        }
        .upload(device);
    }
    let format = match w.dtype.as_str() {
        "fp8_block32" => DeepseekV41WeightFormat::Fp8Block32,
        "fp8_block128" => DeepseekV41WeightFormat::Fp8Block128,
        "fp4_row32" => DeepseekV41WeightFormat::Fp4Row32,
        other => panic!("unknown packed format {other}"),
    };
    DeepseekV41LinearWeights::Packed {
        weight: DeepseekV41PackedWeights::new(
            w.rows,
            w.cols,
            format,
            w.data.clone(),
            w.scales.clone(),
        )?,
        lora,
    }
    .upload(device)
}

/// The released packed/BF16 projection boundary: FP8 activation quantization,
/// block-scaled FP32 dots, one BF16 output rounding, and - with an adapter -
/// the PEFT merge. The quantizer's rounding is a frozen straight-through
/// boundary, so the input gradient contracts against the decoded matrix.
#[test]
fn packed_projection_tape_vjps_match_cpu_pytorch() -> Result<()> {
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41LinearAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in packed_fixture().cases {
        let lora = (c.rank != 0).then(|| DeepseekV41LoraHost {
            rank: c.rank,
            scale: c.scale,
            a: values(&c.a_bits).into(),
            b: values(&c.b_bits).into(),
        });
        let weight = packed_base(&device, &c.weight, lora)?;
        let stream = match c.stream.as_str() {
            "bf16" => DeepseekV41WeightOutput::Bf16,
            "fp32" => DeepseekV41WeightOutput::Fp32,
            other => panic!("unknown stream {other}"),
        };
        let tape = kernels.forward_mixed(&gpu(&device, &c.x_bits)?, c.rows, &weight, stream)?;
        gate(
            &format!("packed {} forward", c.name),
            &tape.output.read_f32(c.rows * c.out)?,
            &c.output_bits,
            &mut failures,
        );
        let gradient = kernels.backward(&tape, &weight, &gpu(&device, &c.upstream_bits)?)?;
        gate(
            &format!("packed {} dX", c.name),
            &gradient.grad_input.read_f32(c.rows * c.inner)?,
            &c.grad_x_bits,
            &mut failures,
        );
        if c.rank != 0 {
            let grad = gradient.read_adapter_gradient("layers.0.ffn.experts.0.w1")?;
            gate(
                &format!("packed {} dA", c.name),
                &grad.grad_a,
                &c.grad_a_bits,
                &mut failures,
            );
            gate(
                &format!("packed {} dB", c.name),
                &grad.grad_b,
                &c.grad_b_bits,
                &mut failures,
            );
        } else {
            assert!(gradient
                .read_adapter_gradient("layers.0.ffn.experts.0.w1")
                .is_err());
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Result<Self> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "v41-projection-step-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        Ok(Self(root))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn adapter(c: &Training, path: &Path) -> Result<DeepseekV41LoraAdapter> {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    fs::create_dir(path)?;
    fs::write(
        path.join("adapter_config.json"),
        serde_json::to_vec_pretty(&c.adapter_config)?,
    )?;
    let mut owned = Vec::new();
    for m in &c.modules {
        for (label, shape, bits) in [
            ("A", vec![m.rank, m.width], &m.a_bits),
            ("B", vec![m.out, m.rank], &m.b_bits),
        ] {
            owned.push((
                format!("base_model.model.{}.lora_{label}.weight", m.target),
                shape,
                bits.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ));
        }
    }
    let views = owned
        .iter()
        .map(|(n, s, b)| Ok((n.as_str(), TensorView::new(Dtype::F32, s.clone(), b)?)))
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, &path.join("adapter_model.safetensors"))?;
    DeepseekV41LoraAdapter::load(path)
}

struct QueryTape {
    first: DeepseekV41LinearTape,
    second: DeepseekV41LinearTape,
    logits: GpuBuffer,
}
fn forward(
    kernels: &DeepseekV41LinearAdjoints,
    c: &Training,
    trainer: &DeepseekV41AdapterTrainer,
    device: &VulkanDevice,
) -> Result<QueryTape> {
    let adapter = trainer.adapter()?;
    let w1 = base(
        device,
        c.latent,
        c.width,
        &c.w1_bits,
        Some(DeepseekV41LoraHost::from_module(
            adapter
                .module_for_tensor("layers.0.attn.wq_a.weight")
                .unwrap(),
        )),
    )?;
    let w2 = base(
        device,
        c.out,
        c.latent,
        &c.w2_bits,
        Some(DeepseekV41LoraHost::from_module(
            adapter
                .module_for_tensor("layers.0.attn.wq_b.weight")
                .unwrap(),
        )),
    )?;
    let first = kernels.forward(&gpu(device, &c.x_bits)?, c.rows, &w1)?;
    let normalized = NormKernel::new(device)?.apply(
        device,
        &first.output,
        &gpu(device, &c.norm_bits)?,
        c.rows,
        c.latent,
        1e-20,
    )?;
    let second = kernels.forward(&normalized, c.rows, &w2)?;
    let logits = LinearKernel::new(device)?.apply(
        device,
        &second.output,
        &gpu(device, &c.head_bits)?,
        c.rows,
        c.out,
        c.vocab,
    )?;
    Ok(QueryTape {
        first,
        second,
        logits,
    })
}
fn train_step(
    kernels: &DeepseekV41LinearAdjoints,
    c: &Training,
    trainer: &mut DeepseekV41AdapterTrainer,
    device: &VulkanDevice,
    expected: &Step,
    failures: &mut Vec<String>,
) -> Result<Vec<f32>> {
    let tape = forward(kernels, c, trainer, device)?;
    gate(
        &format!("{} logits", c.name),
        &tape.logits.read_f32(c.rows * c.vocab)?,
        &expected.logits_bits,
        failures,
    );
    let terminal = DeepseekV41TerminalAdjoints::new(device.clone())?;
    let loss = terminal.cross_entropy_and_input_grad(
        &tape.logits,
        &GpuBuffer::from_u32(device, &c.targets)?,
        &gpu(device, &c.head_bits)?,
        c.rows,
        c.vocab,
        c.out,
    )?;
    if (loss.loss as f64 - expected.loss as f64).abs() >= 2e-7 {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/v41-projection-backward");
        fs::create_dir_all(&root)?;
        fs::write(
            root.join(format!("{}-loss-step{}.json", c.name, trainer.step_count())),
            serde_json::to_vec_pretty(&serde_json::json!({
                "logits":tape.logits.read_f32(c.rows*c.vocab)?,"rows":c.rows,"vocab":c.vocab,"targets":c.targets,
                "row_loss":loss.row_loss,"native_loss":loss.loss,"cpu_loss":expected.loss,
                "cpu_logits_bits":expected.logits_bits,
            }))?,
        )?;
    }
    gate(
        &format!("{} loss", c.name),
        &[loss.loss],
        &[expected.loss.to_bits()],
        failures,
    );
    let w2 = base(device, c.out, c.latent, &c.w2_bits, None)?;
    let second = kernels.backward(&tape.second, &w2, &loss.grad_hidden)?;
    drop(w2);
    let norm_grad = terminal.norm_input_grad(
        &second.grad_input,
        &tape.first.output,
        &gpu(device, &c.norm_bits)?,
        c.rows,
        c.latent,
        1e-20,
    )?;
    let w1 = base(device, c.latent, c.width, &c.w1_bits, None)?;
    let first = kernels.backward(&tape.first, &w1, &norm_grad)?;
    gate(
        &format!("{} chained dX", c.name),
        &first.grad_input.read_f32(c.rows * c.width)?,
        &expected.grad_x_bits,
        failures,
    );
    let gradients = vec![
        first.read_adapter_gradient("layers.0.attn.wq_a")?,
        second.read_adapter_gradient("layers.0.attn.wq_b")?,
    ];
    for grad in &gradients {
        let e = &expected.gradients[&grad.target];
        gate(
            &format!("{} {} dA", c.name, grad.target),
            &grad.grad_a,
            &e.a_bits,
            failures,
        );
        gate(
            &format!("{} {} dB", c.name, grad.target),
            &grad.grad_b,
            &e.b_bits,
            failures,
        );
    }
    trainer.step(&gradients)?;
    for (target, e) in &expected.states {
        let (a, b) = trainer.factors(target).unwrap();
        let (ma, va, mb, vb) = trainer.moments(target).unwrap();
        for (label, actual, expected) in [
            ("A", a, &e.a_bits),
            ("B", b, &e.b_bits),
            ("mA", ma, &e.exp_avg_a_bits),
            ("vA", va, &e.exp_avg_sq_a_bits),
            ("mB", mb, &e.exp_avg_b_bits),
            ("vB", vb, &e.exp_avg_sq_b_bits),
        ] {
            gate(
                &format!("{} {target} {label}", c.name),
                actual,
                expected,
                failures,
            );
        }
    }
    let logits = forward(kernels, c, trainer, device)?
        .logits
        .read_f32(c.rows * c.vocab)?;
    gate(
        &format!("{} updated logits", c.name),
        &logits,
        &expected.logits_after_bits,
        failures,
    );
    Ok(logits)
}

#[test]
fn query_projection_computed_gradients_adamw_resume_export_match_cpu() -> Result<()> {
    let root = Temp::new()?;
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41LinearAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().training {
        let adapter = adapter(&c, &root.0.join(&c.name))?;
        let mut trainer = DeepseekV41AdapterTrainer::new(&adapter, c.hyper)?;
        let mut resumed = None;
        for (step, e) in c.steps.iter().enumerate() {
            let actual = train_step(&kernels, &c, &mut trainer, &device, e, &mut failures)?;
            if step == 1 {
                let checkpoint = root.0.join(format!("{}-resume", c.name));
                trainer.save_checkpoint(&checkpoint)?;
                resumed = Some(DeepseekV41AdapterTrainer::resume_from_checkpoint(
                    &checkpoint,
                )?);
            }
            if step == 2 {
                let resumed = resumed.as_mut().unwrap();
                let restored = train_step(&kernels, &c, resumed, &device, e, &mut failures)?;
                assert_bits(&actual, &restored, "resumed computed-gradient logits");
                assert_eq!(trainer.step_count(), resumed.step_count());
                for target in trainer.targets() {
                    let (a, b) = trainer.factors(target).unwrap();
                    let (ra, rb) = resumed.factors(target).unwrap();
                    assert_bits(a, ra, "resumed A");
                    assert_bits(b, rb, "resumed B");
                    let (ma, va, mb, vb) = trainer.moments(target).unwrap();
                    let (rma, rva, rmb, rvb) = resumed.moments(target).unwrap();
                    for (left, right) in [(ma, rma), (va, rva), (mb, rmb), (vb, rvb)] {
                        assert_bits(left, right, "resumed moments");
                    }
                }
            }
        }
        let exported = root.0.join(format!("{}-trained", c.name));
        trainer.adapter()?.export(&exported)?;
        let loaded = DeepseekV41LoraAdapter::load(&exported)?;
        for target in trainer.targets() {
            let m = loaded
                .module_for_tensor(&format!("{target}.weight"))
                .unwrap();
            let (a, b) = trainer.factors(target).unwrap();
            assert_bits(m.a.as_ref(), a, "exported A");
            assert_bits(m.b.as_ref(), b, "exported B");
        }
        let reloaded = DeepseekV41AdapterTrainer::new(&loaded, c.hyper)?;
        assert_eq!(
            forward(&kernels, &c, &trainer, &device)?
                .logits
                .read_f32(c.rows * c.vocab)?,
            forward(&kernels, &c, &reloaded, &device)?
                .logits
                .read_f32(c.rows * c.vocab)?
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn projection_tape_rejects_unqualified_precision_and_bad_geometry() -> Result<()> {
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41LinearAdjoints::new(device.clone())?;
    let input = GpuBuffer::zeros_f32(&device, 32)?;
    let half = DeepseekV41LinearWeights::Bf16 {
        rows: 4,
        cols: 32,
        values: vec![0.5; 128],
        lora: None,
    }
    .upload(&device)?;
    // The FP32 entry point stays fail-closed for non-FP32 storage; the mixed
    // entry point is the released boundary and rejects only a stream/storage
    // mismatch, exactly as the enclosing graph's cast contract does.
    assert!(kernels.forward(&input, 1, &half).is_err());
    assert!(kernels
        .forward_mixed(&input, 1, &half, DeepseekV41WeightOutput::Fp32)
        .is_err());
    assert!(kernels
        .forward_mixed(&input, 1, &half, DeepseekV41WeightOutput::Bf16)
        .is_ok());
    let packed = DeepseekV41PackedWeights::new(
        32,
        32,
        DeepseekV41WeightFormat::Fp8Block32,
        vec![0; 1024],
        vec![127],
    )?;
    let packed = DeepseekV41LinearWeights::Packed {
        weight: packed,
        lora: None,
    }
    .upload(&device)?;
    // Packed storage quantizes its own activation cast, as the released
    // block-scaled GEMM does, at either stream precision.
    assert!(kernels
        .forward_mixed(&input, 1, &packed, DeepseekV41WeightOutput::Fp32)
        .is_ok());
    assert!(kernels
        .forward_mixed(&input, 1, &packed, DeepseekV41WeightOutput::Bf16)
        .is_ok());
    let dense = base(&device, 4, 32, &vec![0; 128], None)?;
    assert!(kernels
        .forward_mixed(&input, 1, &dense, DeepseekV41WeightOutput::Bf16)
        .is_err());
    assert!(kernels.forward(&input, 0, &dense).is_err());
    let tape = kernels.forward(&input, 1, &dense)?;
    assert!(kernels
        .backward(&tape, &dense, &GpuBuffer::zeros_f32(&device, 1)?)
        .is_err());
    let wrong = base(&device, 5, 32, &vec![0; 160], None)?;
    assert!(kernels.backward(&tape, &wrong, &input).is_err());
    Ok(())
}

#[test]
fn projection_checkpoints_bind_trained_factors_config_and_moments() -> Result<()> {
    let root = Temp::new()?;
    let c = fixture().training.remove(0);
    let original = adapter(&c, &root.0.join("source"))?;
    let trainer = DeepseekV41AdapterTrainer::new(&original, c.hyper)?;
    let checkpoint = root.0.join("checkpoint");
    trainer.save_checkpoint(&checkpoint)?;
    let wrong = original.with_factors(|target| {
        let m = original.module_for_tensor(&format!("{target}.weight"))?;
        let mut a = m.a.to_vec();
        a[0] += 0.01;
        Some((a, m.b.to_vec()))
    })?;
    assert!(DeepseekV41AdapterTrainer::resume(&wrong, &checkpoint).is_err());
    let config = checkpoint.join("adapter_config.json");
    let original_config = fs::read(&config)?;
    let mut changed: serde_json::Value = serde_json::from_slice(&original_config)?;
    changed["use_rslora"] = serde_json::json!(true);
    fs::write(&config, serde_json::to_vec(&changed)?)?;
    assert!(DeepseekV41AdapterTrainer::resume_from_checkpoint(&checkpoint).is_err());
    fs::write(&config, &original_config)?;
    let before = fs::read(checkpoint.join("optimizer_state.safetensors"))?;
    assert!(trainer.save_checkpoint(&checkpoint).is_err());
    assert_eq!(
        before,
        fs::read(checkpoint.join("optimizer_state.safetensors"))?
    );
    let mut corrupt = before.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    fs::write(checkpoint.join("optimizer_state.safetensors"), corrupt)?;
    assert!(DeepseekV41AdapterTrainer::resume_from_checkpoint(&checkpoint).is_err());
    fs::write(checkpoint.join("optimizer_state.safetensors"), before)?;
    let metadata = checkpoint.join("optimizer_state.json");
    let mut old: serde_json::Value = serde_json::from_slice(&fs::read(&metadata)?)?;
    old["schema"] = serde_json::json!(1);
    fs::write(&metadata, serde_json::to_vec(&old)?)?;
    assert!(DeepseekV41AdapterTrainer::resume_from_checkpoint(&checkpoint).is_err());
    Ok(())
}
