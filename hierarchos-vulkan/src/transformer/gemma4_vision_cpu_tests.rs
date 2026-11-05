use super::*;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    case: Case,
}
#[derive(Deserialize)]
struct Case {
    rows: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    grid: usize,
    kernel: usize,
    tokens: Vec<f32>,
    norm_bits: Vec<u32>,
    qn: Vec<f32>,
    rope_bits: Vec<u32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention_bits: Vec<u32>,
    hidden: Vec<f32>,
    pool_bits: Vec<u32>,
    projector_input: Vec<f32>,
    projector_norm_bits: Vec<u32>,
}
fn case() -> Result<Case> {
    let bytes = include_bytes!("../../validation/fixtures/gemma4_cpu_primitives/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "c8728f12a4185d9b117c449fdd5db798c9ffe45e48bdd3d14b0eafbffb9e03ce"
    );
    Ok(serde_json::from_slice::<Fixture>(bytes)?.case)
}
fn exact(label: &str, actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    let diff = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != **b)
        .count();
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - f32::from_bits(*b)).abs())
        .fold(0., f32::max);
    assert_eq!(
        diff,
        0,
        "{label}: {diff}/{} differences, max {worst:e}",
        actual.len()
    );
}
#[test]
fn gemma4_cpu_rms_contiguous_axes_are_word_exact() -> Result<()> {
    let c = case()?;
    let device = VulkanDevice::new()?;
    let kernels = Box::new(TransformerKernels::new(&device)?);
    let norm = VulkanLayerNorm::new_rms_pow(&device, c.dim, 1e-6, &vec![1.; c.dim], 0.)?
        .use_vector_cpu_rsqrt(true);
    for (input, expected) in [
        (&c.tokens, &c.norm_bits),
        (&c.projector_input, &c.projector_norm_bits),
    ] {
        let rows = input.len() / c.dim;
        let x = GpuBuffer::from_f32(&device, input)?;
        let y = GpuBuffer::zeros_f32(&device, input.len())?;
        let mean = GpuBuffer::zeros_f32(&device, rows)?;
        let rstd = GpuBuffer::zeros_f32(&device, rows)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        norm.record_forward(&mut commands, &kernels, &x, &y, &mean, &rstd, rows)?;
        commands.submit()?;
        exact("RMS", &y.read_f32(expected.len())?, expected);
    }
    Ok(())
}
#[test]
fn gemma4_cpu_axial_rope_is_word_exact() -> Result<()> {
    let c = case()?;
    let device = VulkanDevice::new()?;
    let positions = (0..c.grid)
        .flat_map(|y| (0..c.grid).flat_map(move |x| [x as u32, y as u32]))
        .collect::<Vec<_>>();
    let (cos, sin) = rope_tables_gpu(&device, &positions, c.head_dim, 100.)?;
    let x = GpuBuffer::from_f32(&device, &c.qn)?;
    let y = GpuBuffer::zeros_f32(&device, c.qn.len())?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_gemma4_rope.spv"),
        4,
        16,
    )?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[&x, &cos, &sin, &y],
        bytemuck::cast_slice(&[c.rows as u32, c.heads as u32, c.head_dim as u32, 0]),
        [div_ceil_u32(c.qn.len(), 256), 1, 1],
    )?;
    commands.submit()?;
    exact("axial RoPE", &y.read_f32(c.rope_bits.len())?, &c.rope_bits);
    Ok(())
}
fn check_attention(shader: &[u8]) -> Result<()> {
    let c = case()?;
    let device = VulkanDevice::new()?;
    let q = GpuBuffer::from_f32(&device, &c.q)?;
    let k = GpuBuffer::from_f32(&device, &c.k)?;
    let v = GpuBuffer::from_f32(&device, &c.v)?;
    let y = GpuBuffer::zeros_f32(&device, c.attention_bits.len())?;
    let kernel = vulkan::ComputeKernel::new(&device, shader, 4, 16)?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[&q, &k, &v, &y],
        bytemuck::cast_slice(&[1, c.rows as u32, c.dim as u32, c.heads as u32]),
        [c.rows as u32, c.heads as u32, 1],
    )?;
    commands.submit()?;
    exact(
        "attention",
        &y.read_f32(c.attention_bits.len())?,
        &c.attention_bits,
    );
    Ok(())
}
#[test]
fn gemma4_cpu_attention_is_word_exact() -> Result<()> {
    check_attention(include_bytes!(
        "../../shaders/vision_gemma4_attention_cpu_avx2.spv"
    ))
}
#[test]
#[ignore = "external shader diagnostic; not a production math switch"]
fn gemma4_cpu_attention_diagnostic() -> Result<()> {
    check_attention(&std::fs::read(std::env::var(
        "HIERARCHOS_GEMMA4_ATTENTION_SHADER",
    )?)?)
}
#[test]
fn gemma4_cpu_sparse_pool_is_word_exact() -> Result<()> {
    let c = case()?;
    let device = VulkanDevice::new()?;
    let x = GpuBuffer::from_f32(&device, &c.hidden)?;
    let y = GpuBuffer::zeros_f32(&device, c.pool_bits.len())?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_gemma4_pool.spv"),
        2,
        20,
    )?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[&x, &y],
        bytemuck::cast_slice(&[1, c.grid as u32, c.kernel as u32, c.dim as u32, 0]),
        [div_ceil_u32(c.pool_bits.len(), 256), 1, 1],
    )?;
    commands.submit()?;
    exact("pool", &y.read_f32(c.pool_bits.len())?, &c.pool_bits);
    Ok(())
}

#[derive(Deserialize)]
struct AdjointCase {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    dy: Vec<f32>,
    dq_bits: Vec<u32>,
    dk_bits: Vec<u32>,
    dv_bits: Vec<u32>,
}
fn check_adjoint(shader: &[u8]) -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/gemma4_cpu_primitives/adjoint.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "9434734b8704af3fe902b13c18f0d17c878e8db9f7384b1baaf468e1a96190ab"
    );
    let c: AdjointCase = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let q = GpuBuffer::from_f32(&device, &c.q)?;
    let k = GpuBuffer::from_f32(&device, &c.k)?;
    let v = GpuBuffer::from_f32(&device, &c.v)?;
    let dy = GpuBuffer::from_f32(&device, &c.dy)?;
    let dq = GpuBuffer::zeros_f32(&device, c.q.len())?;
    let dk = GpuBuffer::zeros_f32(&device, c.k.len())?;
    let dv = GpuBuffer::zeros_f32(&device, c.v.len())?;
    let kernel = vulkan::ComputeKernel::new(&device, shader, 7, 16)?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[&q, &k, &v, &dy, &dq, &dk, &dv],
        bytemuck::cast_slice(&[1u32, 36, 16, 4]),
        [4, 1, 1],
    )?;
    commands.submit()?;
    for (name, buffer, expected) in [
        ("dQ", &dq, &c.dq_bits),
        ("dK", &dk, &c.dk_bits),
        ("dV", &dv, &c.dv_bits),
    ] {
        exact(name, &buffer.read_f32(expected.len())?, expected);
    }
    Ok(())
}
#[test]
fn gemma4_cpu_attention_adjoint_is_word_exact() -> Result<()> {
    check_adjoint(include_bytes!(
        "../../shaders/vision_gemma4_attention_cpu_avx2_backward.spv"
    ))
}
#[test]
#[ignore = "external shader diagnostic; not a production math switch"]
fn gemma4_cpu_attention_adjoint_diagnostic() -> Result<()> {
    check_adjoint(&std::fs::read(std::env::var(
        "HIERARCHOS_GEMMA4_ATTENTION_BACKWARD_SHADER",
    )?)?)
}

#[derive(Deserialize)]
struct WeightCorpus {
    cases: Vec<WeightCase>,
}
#[derive(Deserialize)]
struct WeightCase {
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input: Vec<f32>,
    gradient: Vec<f32>,
    expected_bits: Vec<u32>,
}
#[test]
fn gemma4_cpu_linear_weight_gradients_are_word_exact() -> Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/gemma4_cpu_primitives/weight-gradient.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "4c910969a1b590e2c30b93ea5a51d18c1f2c6d2ad5b4143be0ff438825378d8d"
    );
    let corpus: WeightCorpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_weight_grad_cpu_avx2.spv"),
        3,
        12,
    )?;
    let mut count = 0;
    for c in corpus.cases {
        let x = GpuBuffer::from_f32(&device, &c.input)?;
        let dy = GpuBuffer::from_f32(&device, &c.gradient)?;
        let dw = GpuBuffer::zeros_f32(&device, c.expected_bits.len())?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&x, &dy, &dw],
            bytemuck::cast_slice(&[c.rows as u32, c.input_dim as u32, c.output_dim as u32]),
            [
                div_ceil_u32(c.input_dim, 16),
                div_ceil_u32(c.output_dim, 16),
                1,
            ],
        )?;
        commands.submit()?;
        exact(
            "linear dW",
            &dw.read_f32(c.expected_bits.len())?,
            &c.expected_bits,
        );
        count += c.expected_bits.len();
    }
    assert_eq!(count, 4032);
    Ok(())
}
#[derive(Deserialize)]
struct TextCase {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    dy: Vec<f32>,
    context_bits: Vec<u32>,
    dq_bits: Vec<u32>,
    dk_bits: Vec<u32>,
    dv_bits: Vec<u32>,
}
#[test]
fn gemma4_cpu_decoder_attention_and_adjoint_are_word_exact() -> Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/gemma4_cpu_primitives/text-attention.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "a04ef9e269d456bbc47885cad2ac5ba6d8f4f776c1a35dd72b5e75e7b1802e4f"
    );
    let c: TextCase = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let q = GpuBuffer::from_f32(&device, &c.q)?;
    let k = GpuBuffer::from_f32(&device, &c.k)?;
    let v = GpuBuffer::from_f32(&device, &c.v)?;
    let dy = GpuBuffer::from_f32(&device, &c.dy)?;
    let output = GpuBuffer::zeros_f32(&device, c.context_bits.len())?;
    let dq = GpuBuffer::zeros_f32(&device, c.q.len())?;
    let dk = GpuBuffer::zeros_f32(&device, c.k.len())?;
    let dv = GpuBuffer::zeros_f32(&device, c.v.len())?;
    let mask = GpuBuffer::from_f32(&device, &[1.; 8])?;
    let dummy = GpuBuffer::zeros_f32(&device, 8)?;
    let mut push = AttentionPush::zeroed();
    push.batch_size = 1;
    push.query_seq_len = 8;
    push.key_value_seq_len = 8;
    push.query_hidden_size = 16;
    push.value_hidden_size = 16;
    push.num_heads = 2;
    push.num_key_value_heads = 2;
    push.scale = 1.;
    push.value_scale = 1.;
    push.causal = 1;
    push.unfused_qk = 1;
    push.kv_page_size = 1;
    push.kv_page_table_stride = 1;
    let forward = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_attention_forward_gemma3_cpu_avx2.spv"),
        11,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let backward = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_attention_backward_gemma4_cpu_avx2.spv"),
        15,
        std::mem::size_of::<AttentionBackwardPush>() as u32,
    )?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    forward.record_dispatch(
        &mut commands,
        &[
            &q, &k, &v, &mask, &mask, &dummy, &output, &dummy, &dummy, &mask, &mask,
        ],
        bytemuck::bytes_of(&push),
        [8, 2, 1],
    )?;
    backward.record_dispatch(
        &mut commands,
        &[
            &q, &k, &v, &mask, &mask, &dummy, &dy, &dq, &dk, &dv, &dummy, &dummy, &dummy, &dummy,
            &mask,
        ],
        bytemuck::bytes_of(&push.as_backward_push(0, 1)),
        [2, 1, 1],
    )?;
    commands.submit()?;
    for (name, buffer, expected) in [
        ("decoder attention", &output, &c.context_bits),
        ("decoder dQ", &dq, &c.dq_bits),
        ("decoder dK", &dk, &c.dk_bits),
        ("decoder dV", &dv, &c.dv_bits),
    ] {
        exact(name, &buffer.read_f32(expected.len())?, expected);
    }
    Ok(())
}
