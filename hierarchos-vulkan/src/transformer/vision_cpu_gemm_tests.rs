use super::*;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct OnednnGeluCorpus {
    input: Vec<f32>,
    gradient: Vec<f32>,
    expected_forward_bits: Vec<u32>,
    expected_backward_bits: Vec<u32>,
}

#[test]
fn vision_onednn_gelu_matches_frozen_active_cpu_dispatch() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/vision_avx2_row_tail/onednn_gelu.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "b06af3324061285e28151bbecde75dc150f1be994f2f68489726a6c283689c3c"
    );
    let fixture: OnednnGeluCorpus = serde_json::from_slice(bytes)?;
    let n = fixture.input.len();
    assert_eq!(n, 4616);
    let device = VulkanDevice::new()?;
    let input = GpuBuffer::from_f32(&device, &fixture.input)?;
    let gradient = GpuBuffer::from_f32(&device, &fixture.gradient)?;
    let output = GpuBuffer::zeros_f32(&device, n)?;
    for backward in [false, true] {
        let code: &[u8] = if backward {
            include_bytes!("../../shaders/vision_gelu_erf_cpu_backward.spv")
        } else {
            include_bytes!("../../shaders/vision_gelu_erf_cpu_forward.spv")
        };
        let kernel = vulkan::ComputeKernel::new(&device, code, if backward { 3 } else { 2 }, 4)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        let bindings = if backward {
            vec![&input, &gradient, &output]
        } else {
            vec![&input, &output]
        };
        kernel.record_dispatch(
            &mut commands,
            &bindings,
            bytemuck::cast_slice(&[n as u32]),
            [n.div_ceil(256) as u32, 1, 1],
        )?;
        commands.submit()?;
        let actual = output.read_f32(n)?;
        let expected = if backward {
            &fixture.expected_backward_bits
        } else {
            &fixture.expected_forward_bits
        };
        let differences = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| a.to_bits() != **b)
            .count();
        assert_eq!(differences, 0, "oneDNN GELU backward={backward}");
    }
    Ok(())
}

#[derive(Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    expected_bits: Vec<u32>,
}

#[test]
fn vision_cpu_patch_convolution_dispatch_is_word_exact() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/vision_avx2_row_tail/patch.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "27dcb7d7c87d4627378dc9d1ed00990da51a3f3f40d080ea42e34f28a1c14ff2"
    );
    let corpus: Corpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_patch_projection.spv"),
        4,
        16,
    )?;
    let mut count = 0;
    for case in corpus.cases {
        let input = GpuBuffer::from_f32(&device, &case.input)?;
        let weight = GpuBuffer::from_f32(&device, &case.weight)?;
        let bias = GpuBuffer::from_f32(&device, &case.bias)?;
        let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
        let push = [
            case.rows as u32,
            case.input_dim as u32,
            case.output_dim as u32,
            u32::from(case.rows == 32 || case.input_dim == 48),
        ];
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &weight, &bias, &output],
            bytemuck::cast_slice(&push),
            [
                div_ceil_u32(case.output_dim, 16),
                div_ceil_u32(case.rows, 16),
                1,
            ],
        )?;
        commands.submit()?;
        let actual = output.read_f32(case.expected_bits.len())?;
        let differences = actual
            .iter()
            .zip(&case.expected_bits)
            .filter(|(a, b)| a.to_bits() != **b)
            .count();
        assert_eq!(
            differences, 0,
            "patch M={} K={} N={}",
            case.rows, case.input_dim, case.output_dim
        );
        count += actual.len();
    }
    assert_eq!(count, 896);
    Ok(())
}

#[test]
fn vision_cpu_gemm_limits_admission_to_measured_shapes() {
    assert!(cpu_avx2_vision_gemm(32, 16, 16, 8));
    assert!(cpu_avx2_vision_gemm(32, 12, 16, 8));
    for (rows, input, output, width) in [
        (1, 16, 16, 8),
        (37, 16, 16, 8),
        (32, 6, 16, 8),
        (32, 64, 16, 8),
        (32, 16, 8, 8),
        (32, 16, 16, 16),
    ] {
        assert!(!cpu_avx2_vision_gemm(rows, input, output, width));
    }
}

#[test]
fn vision_cpu_gemm_frozen_panels_and_tails_are_word_exact() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/vision_avx2_row_tail/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "3a88289d8af3fb628cb735ca3b6889e01ed82a1ced38b018b444ddc35ae6017f"
    );
    let corpus: Corpus = serde_json::from_slice(bytes)?;
    assert_eq!(corpus.cases.len(), 10);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
        4,
        16,
    )?;
    let mut count = 0;
    for case in corpus.cases {
        assert!(cpu_avx2_vision_gemm(
            case.rows,
            case.input_dim,
            case.output_dim,
            8
        ));
        let input = GpuBuffer::from_f32(&device, &case.input)?;
        let weight = GpuBuffer::from_f32(&device, &case.weight)?;
        let bias = GpuBuffer::from_f32(&device, &case.bias)?;
        let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
        let push = [
            case.rows as u32,
            case.input_dim as u32,
            case.output_dim as u32,
            1,
        ];
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &weight, &bias, &output],
            bytemuck::cast_slice(&push),
            [
                div_ceil_u32(case.output_dim, 16),
                div_ceil_u32(case.rows, 16),
                1,
            ],
        )?;
        commands.submit()?;
        let actual = output.read_f32(case.expected_bits.len())?;
        let differences = actual
            .iter()
            .zip(&case.expected_bits)
            .filter(|(a, b)| a.to_bits() != **b)
            .count();
        assert_eq!(
            differences,
            0,
            "M={} K={} N={} on {}",
            case.rows,
            case.input_dim,
            case.output_dim,
            device.name()
        );
        count += actual.len();
    }
    assert_eq!(count, 4208);
    eprintln!(
        "vision AVX2 GEMM: {count} frozen CPU words exact on {}",
        device.name()
    );
    Ok(())
}

#[derive(Deserialize)]
struct AttentionCorpus {
    images: u32,
    seq: u32,
    dim: u32,
    heads: u32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    expected_bits: Vec<u32>,
}

#[test]
fn vision_cpu_attention_frozen_context_is_word_exact() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/vision_avx2_row_tail/attention.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "0ec6389d249d019c5937141a6f6b105fd6280c7cb4bf05e7553ce33cdab364ce"
    );
    let case: AttentionCorpus = serde_json::from_slice(bytes)?;
    assert_eq!(case.expected_bits.len(), 512);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_attention_cpu_avx2.spv"),
        4,
        16,
    )?;
    let q = GpuBuffer::from_f32(&device, &case.q)?;
    let k = GpuBuffer::from_f32(&device, &case.k)?;
    let v = GpuBuffer::from_f32(&device, &case.v)?;
    let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[&q, &k, &v, &output],
        bytemuck::cast_slice(&[case.images, case.seq, case.dim, case.heads]),
        [case.seq, case.heads, case.images],
    )?;
    commands.submit()?;
    let actual = output.read_f32(case.expected_bits.len())?;
    let diff_indices: Vec<usize> = actual
        .iter()
        .zip(&case.expected_bits)
        .enumerate()
        .filter_map(|(i, (a, b))| (a.to_bits() != *b).then_some(i))
        .collect();
    for &i in &diff_indices {
        eprintln!(
            "seq5 diff[{i}] actual={} ({:#010x}) expected={} ({:#010x})",
            actual[i],
            actual[i].to_bits(),
            f32::from_bits(case.expected_bits[i]),
            case.expected_bits[i]
        );
    }
    eprintln!("seq5 diff indices: {:?}", diff_indices);
    let differences = diff_indices.len();
    assert_eq!(differences, 0, "SigLIP CPU attention on {}", device.name());
    eprintln!(
        "vision AVX2 attention: 512 frozen CPU context words exact on {}",
        device.name()
    );
    Ok(())
}

#[derive(Deserialize)]
struct InputGradientCorpus {
    cases: Vec<InputGradientCase>,
}

#[derive(Deserialize)]
struct InputGradientCase {
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input: Vec<f32>,
    weight_transposed: Vec<f32>,
    expected_bits: Vec<u32>,
}

#[test]
fn vision_cpu_input_gradient_frozen_panels_and_tails_are_word_exact() -> Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/input-gradient.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "2a8ffe83acaabb6445600b75cf020757590e5be57262b36bfd1dc20d0a10205a"
    );
    let mut corpus: InputGradientCorpus = serde_json::from_slice(bytes)?;
    let wide =
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/input-gradient-wide.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(wide)),
        "57f0ec22b6471a1f79653199e7d596caa622f48c7b84ae21dfec7f4c7be9ac14"
    );
    let extra: InputGradientCorpus = serde_json::from_slice(wide)?;
    corpus.cases.extend(extra.cases);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv"),
        4,
        16,
    )?;
    let dummy = GpuBuffer::from_f32(&device, &[0.0])?;
    let mut count = 0;
    for case in corpus.cases {
        let input = GpuBuffer::from_f32(&device, &case.input)?;
        let weight = GpuBuffer::from_f32(&device, &case.weight_transposed)?;
        let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &weight, &dummy, &output],
            bytemuck::cast_slice(&[
                case.rows as u32,
                case.input_dim as u32,
                case.output_dim as u32,
                0,
            ]),
            [
                div_ceil_u32(case.output_dim, 16),
                div_ceil_u32(case.rows, 16),
                1,
            ],
        )?;
        commands.submit()?;
        let actual = output.read_f32(case.expected_bits.len())?;
        let differences = actual
            .iter()
            .zip(&case.expected_bits)
            .filter(|(a, b)| a.to_bits() != **b)
            .count();
        assert_eq!(
            differences, 0,
            "input-gradient M={} K={} N={}",
            case.rows, case.input_dim, case.output_dim
        );
        count += actual.len();
    }
    assert_eq!(count, 5632);
    eprintln!(
        "vision AVX2 input gradient: {count} frozen CPU words exact on {}",
        device.name()
    );
    Ok(())
}

#[derive(Deserialize)]
struct NormCorpus {
    cases: Vec<NormCase>,
}

#[derive(Deserialize)]
struct NormCase {
    rows: usize,
    dim: usize,
    x: Vec<f32>,
    gamma: Vec<f32>,
    dy: Vec<f32>,
    mean: Vec<f32>,
    rstd: Vec<f32>,
    dx_bits: Vec<u32>,
    dg_bits: Vec<u32>,
    db_bits: Vec<u32>,
}

#[test]
fn vision_cpu_layer_norm_backward_is_word_exact() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/vision_avx2_row_tail/norm-backward.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "b0fbc784dde48ccdc069d4f71eba662f10415d3031a5d078d01f9b2cd515648d"
    );
    let corpus: NormCorpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let input_kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_layer_norm_input_grad_cpu_avx2.spv"),
        6,
        8,
    )?;
    let param_kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_layer_norm_param_grad_cpu_avx2.spv"),
        6,
        8,
    )?;
    let mut count = 0;
    for case in corpus.cases {
        let x = GpuBuffer::from_f32(&device, &case.x)?;
        let gamma = GpuBuffer::from_f32(&device, &case.gamma)?;
        let dy = GpuBuffer::from_f32(&device, &case.dy)?;
        let mean = GpuBuffer::from_f32(&device, &case.mean)?;
        let rstd = GpuBuffer::from_f32(&device, &case.rstd)?;
        let dx = GpuBuffer::zeros_f32(&device, case.dx_bits.len())?;
        let dg = GpuBuffer::zeros_f32(&device, case.dim)?;
        let db = GpuBuffer::zeros_f32(&device, case.dim)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        let push = [case.rows as u32, case.dim as u32];
        input_kernel.record_dispatch(
            &mut commands,
            &[&dy, &x, &gamma, &mean, &rstd, &dx],
            bytemuck::cast_slice(&push),
            [div_ceil_u32(case.rows, 64), 1, 1],
        )?;
        param_kernel.record_dispatch(
            &mut commands,
            &[&dy, &x, &mean, &rstd, &dg, &db],
            bytemuck::cast_slice(&push),
            [div_ceil_u32(case.dim, 64), 1, 1],
        )?;
        commands.submit()?;
        for (name, buffer, expected) in [
            ("dx", &dx, &case.dx_bits),
            ("dg", &dg, &case.dg_bits),
            ("db", &db, &case.db_bits),
        ] {
            let actual = buffer.read_f32(expected.len())?;
            let differences = actual
                .iter()
                .zip(expected)
                .filter(|(a, b)| a.to_bits() != **b)
                .count();
            assert_eq!(
                differences, 0,
                "LayerNorm {name} M={} N={}",
                case.rows, case.dim
            );
            count += actual.len();
        }
    }
    assert_eq!(count, 1600);
    eprintln!(
        "vision AVX2 LayerNorm backward: {count} frozen CPU words exact on {}",
        device.name()
    );
    Ok(())
}

#[derive(Deserialize)]
struct UnaryCorpus {
    input: Vec<f32>,
    expected_bits: Vec<u32>,
}

fn check_unary_cpu_words(shader: &[u8], bytes: &[u8]) -> Result<()> {
    let case: UnaryCorpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(&device, shader, 2, 4)?;
    let input = GpuBuffer::from_f32(&device, &case.input)?;
    let output = GpuBuffer::zeros_f32(&device, case.input.len())?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[&input, &output],
        bytemuck::bytes_of(&(case.input.len() as u32)),
        [div_ceil_u32(case.input.len(), 256), 1, 1],
    )?;
    commands.submit()?;
    let actual = output.read_f32(case.input.len())?;
    assert_eq!(actual.len(), case.expected_bits.len());
    let differences = actual
        .iter()
        .zip(&case.expected_bits)
        .filter(|(a, b)| a.to_bits() != **b)
        .count();
    let worst = actual
        .iter()
        .zip(&case.expected_bits)
        .map(|(a, b)| (*a - f32::from_bits(*b)).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "unary CPU replay: {differences}/{} word differences, max {worst:e}",
        actual.len()
    );
    assert_eq!(differences, 0);
    Ok(())
}

#[test]
fn vision_cpu_builtin_gelu_is_word_exact() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/vision_avx2_row_tail/gelu.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "a85d6ce5f0588d1327f14fae675f96ed4e1c71edd4735a18c792582ca63edb8a"
    );
    check_unary_cpu_words(
        include_bytes!("../../shaders/transformer_gelu_tanh_forward.spv"),
        bytes,
    )
}

#[test]
#[ignore = "opt-in standalone SPIR-V unary diagnostic; never a pipeline acceptance gate"]
fn vision_unary_shader_diagnostic() -> Result<()> {
    let shader = fs::read(std::env::var("HIERARCHOS_VISION_UNARY_SHADER")?)?;
    let fixture = fs::read(std::env::var("HIERARCHOS_VISION_UNARY_FIXTURE")?)?;
    check_unary_cpu_words(&shader, &fixture)
}

#[derive(Deserialize)]
struct ProjectorNormCorpus {
    rows: usize,
    dim: usize,
    input: Vec<f32>,
    gradient: Vec<f32>,
    expected_bits: Vec<u32>,
    dx_bits: Vec<u32>,
}

#[test]
fn vision_cpu_projector_norm_matches_strided_forward_and_contiguous_adjoint() -> Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/projector-norm.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "7170f2a1a75d55e7d1c695989116f52669399615c48f24c403656d9f183bf0b8"
    );
    let case: ProjectorNormCorpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernels = Box::new(TransformerKernels::new(&device)?);
    let norm = VulkanLayerNorm::new_rms(&device, case.dim, 1e-6, &vec![0.; case.dim], 1.)?
        .use_unfused_rms_products(true)
        .use_torch_cpu_rsqrt(true);
    let input = GpuBuffer::from_f32(&device, &case.input)?;
    let gradient = GpuBuffer::from_f32(&device, &case.gradient)?;
    let output = GpuBuffer::zeros_f32(&device, case.input.len())?;
    let dx = GpuBuffer::zeros_f32(&device, case.input.len())?;
    let mean = GpuBuffer::zeros_f32(&device, case.rows)?;
    let rstd = GpuBuffer::zeros_f32(&device, case.rows)?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    norm.record_forward(
        &mut commands,
        &kernels,
        &input,
        &output,
        &mean,
        &rstd,
        case.rows,
    )?;
    norm.record_backward(
        &mut commands,
        &kernels,
        &input,
        &gradient,
        &mean,
        &rstd,
        &dx,
        case.rows,
    )?;
    commands.submit()?;
    for (label, buffer, expected) in [
        ("forward", &output, &case.expected_bits),
        ("adjoint", &dx, &case.dx_bits),
    ] {
        let actual = buffer.read_f32(expected.len())?;
        let differences = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| a.to_bits() != **b)
            .count();
        assert_eq!(differences, 0, "projector RMSNorm {label}");
    }
    eprintln!("projector RMSNorm: 256 frozen CPU forward/adjoint words exact");
    Ok(())
}

#[derive(Deserialize)]
struct TextAttentionCorpus {
    cases: Vec<TextAttentionCase>,
}

#[derive(Deserialize)]
struct TextAttentionCase {
    #[serde(flatten)]
    attention: AttentionCorpus,
    mask: Vec<f32>,
    visible: Vec<bool>,
}

#[test]
fn vision_cpu_composed_text_attention_matches_frozen_eager_contexts() -> Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/text-attention.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "3d6d6178726f0eec4d8ecf5e2f2763ae3b30087046c9cbc7940aa901ff7b6638"
    );
    let corpus: TextAttentionCorpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_attention_forward_gemma3_cpu_avx2.spv"),
        11,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let dummy = GpuBuffer::from_f32(&device, &[0.])?;
    let mut count = 0;
    for case in corpus.cases {
        let a = case.attention;
        let q = GpuBuffer::from_f32(&device, &a.q)?;
        let k = GpuBuffer::from_f32(&device, &a.k)?;
        let v = GpuBuffer::from_f32(&device, &a.v)?;
        let mask = GpuBuffer::from_f32(&device, &case.mask)?;
        let output = GpuBuffer::zeros_f32(&device, a.expected_bits.len())?;
        let mut push = AttentionPush::zeroed();
        push.batch_size = a.images;
        push.query_seq_len = a.seq;
        push.key_value_seq_len = a.seq;
        push.query_hidden_size = a.dim;
        push.value_hidden_size = a.dim;
        push.num_heads = a.heads;
        push.num_key_value_heads = a.heads;
        push.scale = (8_f64.sqrt().recip()) as f32;
        push.value_scale = 1.;
        push.causal = 1;
        push.unfused_qk = 9;
        push.kv_page_size = 1;
        push.kv_page_table_stride = 1;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[
                &q, &k, &v, &mask, &mask, &dummy, &output, &dummy, &dummy, &dummy, &dummy,
            ],
            bytemuck::bytes_of(&push),
            [a.seq, a.heads, a.images],
        )?;
        commands.submit()?;
        let actual = output.read_f32(a.expected_bits.len())?;
        assert!(actual.iter().all(|x| x.is_finite()));
        let differences = actual
            .iter()
            .zip(&a.expected_bits)
            .zip(&case.visible)
            .filter(|((x, y), visible)| **visible && x.to_bits() != **y)
            .count();
        assert_eq!(
            differences, 0,
            "composed text attention seq={} images={}",
            a.seq, a.images
        );
        count += case.visible.iter().filter(|&&v| v).count();
    }
    assert_eq!(count, 704);
    eprintln!("composed Gemma3 attention: {count} frozen CPU context words exact");
    Ok(())
}

#[derive(Deserialize)]
struct Seq5AttentionCorpus {
    images: u32,
    seq: u32,
    dim: u32,
    heads: u32,
    scale: f32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    #[serde(deserialize_with = "deserialize_fp32_bits")]
    expected_bits: Vec<u32>,
}

#[test]
fn vision_cpu_seq5_attention_matches_frozen_eager_context() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/seq5_attention.json");
    let case: Seq5AttentionCorpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/seq5_attention_cpu_avx2.spv"),
        4,
        20,
    )?;
    let q = GpuBuffer::from_f32(&device, &case.q)?;
    let k = GpuBuffer::from_f32(&device, &case.k)?;
    let v = GpuBuffer::from_f32(&device, &case.v)?;
    let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    let push = [
        case.images,
        case.seq,
        case.dim,
        case.heads,
        case.scale.to_bits(),
    ];
    kernel.record_dispatch(
        &mut commands,
        &[&q, &k, &v, &output],
        bytemuck::cast_slice(&push),
        [case.seq, case.heads, case.images],
    )?;
    commands.submit()?;
    let actual = output.read_f32(case.expected_bits.len())?;
    let differences = actual
        .iter()
        .zip(&case.expected_bits)
        .filter(|(a, b)| a.to_bits() != **b)
        .count();
    let worst = actual
        .iter()
        .zip(&case.expected_bits)
        .map(|(a, b)| (*a - f32::from_bits(*b)).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "seq5 CPU attention: {differences}/{} differences, max {worst:e} on {}",
        actual.len(),
        device.name()
    );
    assert_eq!(differences, 0, "seq5 CPU attention differs");
    Ok(())
}

#[derive(Deserialize)]
struct ImageTextAttentionCorpus {
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale: f32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    #[serde(deserialize_with = "deserialize_fp32_bits")]
    expected_bits: Vec<u32>,
}

// These two historical Torch fixtures store view(int32) words as signed JSON
// integers. Preserve their bytes and reinterpret that original representation.
fn deserialize_fp32_bits<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<u32>, D::Error> {
    Vec::<i64>::deserialize(deserializer)?
        .into_iter()
        .map(|word| {
            u32::try_from(word)
                .or_else(|_| i32::try_from(word).map(|word| word as u32))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

#[test]
fn image_text_cpu_attention_matches_frozen_eager_context() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/image_text_attention_gpt2.json");
    let case: ImageTextAttentionCorpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/image_text_attention_cpu_avx2.spv"),
        11,
        30 * 4,
    )?;
    let q = GpuBuffer::from_f32(&device, &case.q)?;
    let k = GpuBuffer::from_f32(&device, &case.k)?;
    let v = GpuBuffer::from_f32(&device, &case.v)?;
    let query_mask = GpuBuffer::from_f32(&device, &vec![1.0; (case.batch * case.seq) as usize])?;
    let key_mask = GpuBuffer::from_f32(&device, &vec![1.0; (case.batch * case.seq) as usize])?;
    let alibi = GpuBuffer::zeros_f32(&device, 1)?;
    let sink = GpuBuffer::zeros_f32(&device, 1)?;
    let relative = GpuBuffer::zeros_f32(&device, 1)?;
    let sparse = GpuBuffer::zeros_f32(&device, 1)?;
    let page = GpuBuffer::zeros_u32(&device, 1)?;
    let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
    let query_hidden = case.heads * case.head_dim;
    let value_hidden = case.kv_heads * case.head_dim;
    let push = [
        case.batch,
        case.seq,
        case.seq,
        query_hidden,
        value_hidden,
        case.heads,
        case.kv_heads,
        0,
        case.scale.to_bits(),
        1.0f32.to_bits(),
        0.0f32.to_bits(),
        0,
        8.0f32.to_bits(),
        0.0f32.to_bits(),
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[
            &q,
            &k,
            &v,
            &query_mask,
            &key_mask,
            &alibi,
            &output,
            &sink,
            &relative,
            &sparse,
            &page,
        ],
        bytemuck::cast_slice(&push),
        [case.seq, case.heads, case.batch],
    )?;
    commands.submit()?;
    let actual = output.read_f32(case.expected_bits.len())?;
    let differences = actual
        .iter()
        .zip(&case.expected_bits)
        .filter(|(a, b)| a.to_bits() != **b)
        .count();
    let worst = actual
        .iter()
        .zip(&case.expected_bits)
        .map(|(a, b)| (*a - f32::from_bits(*b)).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "image-text CPU attention: {differences}/{} differences, max {worst:e} on {}",
        actual.len(),
        device.name()
    );
    assert_eq!(differences, 0, "image-text CPU attention differs");
    Ok(())
}

// --- RDNA3 staged multimodal schedules -------------------------------------
//
// The RDNA3 modules keep the exact per-element arithmetic of the committed
// vision kernels and change only where the operands are read from. These tests
// execute both modules on the running adapter and require raw word identity;
// the timings are recorded for the RDNA3 port receipt, not as a cross-adapter
// performance claim.

fn dispatch_words(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    base_bindings: &[&GpuBuffer],
    output: &GpuBuffer,
    push: &[u8],
    groups: [u32; 3],
    len: usize,
) -> Result<Vec<f32>> {
    let mut bindings = base_bindings.to_vec();
    bindings.push(output);
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(&mut commands, &bindings, push, groups)?;
    commands.submit()?;
    output.read_f32(len)
}

fn median_dispatch_ms(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    base_bindings: &[&GpuBuffer],
    output: &GpuBuffer,
    push: &[u8],
    groups: [u32; 3],
    len: usize,
    iterations: usize,
) -> Result<f64> {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut bindings = base_bindings.to_vec();
        bindings.push(output);
        let mut commands = vulkan::ComputeBatch::new(device)?;
        let start = std::time::Instant::now();
        kernel.record_dispatch(&mut commands, &bindings, push, groups)?;
        commands.submit()?;
        let _ = output.read_f32(len)?;
        samples.push(start.elapsed().as_secs_f64() * 1.0e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timing samples"));
    Ok(samples[samples.len() / 2])
}

fn assert_words_equal(label: &str, staged: &[f32], portable: &[f32]) -> Result<()> {
    if staged.len() != portable.len() {
        anyhow::bail!(
            "{label}: staged length {} != portable {}",
            staged.len(),
            portable.len()
        );
    }
    let differences = staged
        .iter()
        .zip(portable)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differences, 0, "{label}: {differences} differing words");
    Ok(())
}

#[test]
fn rdna3_hw_fma_vision_linear_is_word_exact_and_measured() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/vision_avx2_row_tail/fixture.json");
    let corpus: Corpus = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let portable = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
        4,
        16,
    )?;
    let staged = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_hw_fma_amd_rdna3.spv"),
        4,
        16,
    )?;
    let mut words = 0usize;
    let mut timed = false;
    for case in &corpus.cases {
        assert!(cpu_avx2_vision_gemm(
            case.rows,
            case.input_dim,
            case.output_dim,
            8
        ));
        let input = GpuBuffer::from_f32(&device, &case.input)?;
        let weight = GpuBuffer::from_f32(&device, &case.weight)?;
        let bias = GpuBuffer::from_f32(&device, &case.bias)?;
        let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
        let push = [
            case.rows as u32,
            case.input_dim as u32,
            case.output_dim as u32,
            1,
        ];
        let groups = [
            div_ceil_u32(case.output_dim, 16),
            div_ceil_u32(case.rows, 16),
            1,
        ];
        let bindings = [&input, &weight, &bias];
        let reference = dispatch_words(
            &device,
            &portable,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            case.expected_bits.len(),
        )?;
        let candidate = dispatch_words(
            &device,
            &staged,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            case.expected_bits.len(),
        )?;
        assert_words_equal(
            &format!(
                "RDNA3 vision linear M={} K={} N={}",
                case.rows, case.input_dim, case.output_dim
            ),
            &candidate,
            &reference,
        )?;
        words += reference.len();
        if !timed {
            let portable_ms = median_dispatch_ms(
                &device,
                &portable,
                &bindings,
                &output,
                bytemuck::cast_slice(&push),
                groups,
                case.expected_bits.len(),
                20,
            )?;
            let staged_ms = median_dispatch_ms(
                &device,
                &staged,
                &bindings,
                &output,
                bytemuck::cast_slice(&push),
                groups,
                case.expected_bits.len(),
                20,
            )?;
            eprintln!(
                "RDNA3 vision linear M={} K={} N={}: portable {portable_ms:.4} ms, hardware {staged_ms:.4} ms, ratio {:.3}x",
                case.rows, case.input_dim, case.output_dim, portable_ms / staged_ms
            );
            timed = true;
        }
    }
    assert_eq!(words, 4208);
    eprintln!(
        "RDNA3 hardware FMA vision linear: {words} frozen words exact on {}",
        device.name()
    );
    Ok(())
}

#[test]
fn rdna3_hw_fma_vision_input_gradient_is_word_exact_and_measured() -> Result<()> {
    let mut corpus: InputGradientCorpus = serde_json::from_slice(include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/input-gradient.json"
    ))?;
    let extra: InputGradientCorpus = serde_json::from_slice(include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/input-gradient-wide.json"
    ))?;
    corpus.cases.extend(extra.cases);
    let device = VulkanDevice::new()?;
    let portable = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv"),
        4,
        16,
    )?;
    let staged = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_input_grad_hw_fma_amd_rdna3.spv"),
        4,
        16,
    )?;
    let dummy = GpuBuffer::from_f32(&device, &[0.0])?;
    let mut words = 0usize;
    let mut timed = false;
    for case in &corpus.cases {
        let gradient = GpuBuffer::from_f32(&device, &case.input)?;
        let weight = GpuBuffer::from_f32(&device, &case.weight_transposed)?;
        let output = GpuBuffer::zeros_f32(&device, case.expected_bits.len())?;
        let push = [
            case.rows as u32,
            case.input_dim as u32,
            case.output_dim as u32,
            0,
        ];
        let groups = [
            div_ceil_u32(case.output_dim, 16),
            div_ceil_u32(case.rows, 16),
            1,
        ];
        let bindings = [&gradient, &weight, &dummy];
        let reference = dispatch_words(
            &device,
            &portable,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            case.expected_bits.len(),
        )?;
        let candidate = dispatch_words(
            &device,
            &staged,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            case.expected_bits.len(),
        )?;
        assert_words_equal(
            &format!(
                "RDNA3 vision dX M={} K={} N={}",
                case.rows, case.input_dim, case.output_dim
            ),
            &candidate,
            &reference,
        )?;
        words += reference.len();
        if !timed {
            let portable_ms = median_dispatch_ms(
                &device,
                &portable,
                &bindings,
                &output,
                bytemuck::cast_slice(&push),
                groups,
                case.expected_bits.len(),
                20,
            )?;
            let staged_ms = median_dispatch_ms(
                &device,
                &staged,
                &bindings,
                &output,
                bytemuck::cast_slice(&push),
                groups,
                case.expected_bits.len(),
                20,
            )?;
            eprintln!(
                "RDNA3 vision dX M={} K={} N={}: portable {portable_ms:.4} ms, hardware {staged_ms:.4} ms, ratio {:.3}x",
                case.rows, case.input_dim, case.output_dim, portable_ms / staged_ms
            );
            timed = true;
        }
    }
    assert!(words >= 5632);
    eprintln!(
        "RDNA3 hardware FMA vision dX: {words} frozen words exact on {}",
        device.name()
    );
    Ok(())
}

fn deterministic_values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = (index * 37 + salt * 53 + 11) % 251;
            (value as f32 - 125.0) / 113.0
        })
        .collect()
}

#[test]
fn rdna3_hw_fma_patch_linear_is_word_exact() -> Result<()> {
    let device = VulkanDevice::new()?;
    let portable = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_patch_linear_cpu_avx2.spv"),
        3,
        16,
    )?;
    let staged = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_patch_linear_hw_fma_amd_rdna3.spv"),
        3,
        16,
    )?;
    let mut words = 0usize;
    let geometries: Vec<(usize, usize, usize, usize)> = [2usize, 3, 4, 5, 6, 7, 8, 12, 32, 36]
        .into_iter()
        .flat_map(|rows| [(48usize, 16usize), (12, 16), (48, 32)].map(|(k, n)| (rows, k, n, 3)))
        .collect();
    for (rows, input_dim, output_dim, channels) in geometries {
        let input = GpuBuffer::from_f32(&device, &deterministic_values(rows * input_dim, 1))?;
        let weight =
            GpuBuffer::from_f32(&device, &deterministic_values(output_dim * input_dim, 2))?;
        let output = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let push = PatchLinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            channels: channels as u32,
        };
        let groups = [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1];
        let bindings = [&input, &weight];
        let reference = dispatch_words(
            &device,
            &portable,
            &bindings,
            &output,
            bytemuck::bytes_of(&push),
            groups,
            rows * output_dim,
        )?;
        let candidate = dispatch_words(
            &device,
            &staged,
            &bindings,
            &output,
            bytemuck::bytes_of(&push),
            groups,
            rows * output_dim,
        )?;
        assert_words_equal(
            &format!("RDNA3 patch linear M={rows} K={input_dim} N={output_dim}"),
            &candidate,
            &reference,
        )?;
        words += reference.len();
    }
    eprintln!(
        "RDNA3 hardware FMA patch linear: {words} words exact on {}",
        device.name()
    );
    Ok(())
}

#[test]
fn rdna3_hw_fma_weight_gradient_is_word_exact_and_measured() -> Result<()> {
    let device = VulkanDevice::new()?;
    let portable = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_weight_grad.spv"),
        3,
        12,
    )?;
    let staged = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_weight_grad_hw_fma_amd_rdna3.spv"),
        3,
        12,
    )?;
    let mut words = 0usize;
    for (rows, input_dim, output_dim) in [
        (36usize, 16usize, 16usize),
        (16, 16, 16),
        (8, 12, 16),
        (7, 16, 32),
        (4, 16, 16),
    ] {
        let input = GpuBuffer::from_f32(&device, &deterministic_values(rows * input_dim, 3))?;
        let gradient = GpuBuffer::from_f32(&device, &deterministic_values(rows * output_dim, 4))?;
        let output = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let push = [rows as u32, input_dim as u32, output_dim as u32];
        let groups = [div_ceil_u32(input_dim, 16), div_ceil_u32(output_dim, 16), 1];
        let bindings = [&input, &gradient];
        let reference = dispatch_words(
            &device,
            &portable,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            output_dim * input_dim,
        )?;
        let candidate = dispatch_words(
            &device,
            &staged,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            output_dim * input_dim,
        )?;
        assert_words_equal(
            &format!("RDNA3 weight grad M={rows} K={input_dim} N={output_dim}"),
            &candidate,
            &reference,
        )?;
        words += reference.len();
        if rows == 36 && input_dim == 16 && output_dim == 16 {
            let portable_ms = median_dispatch_ms(
                &device,
                &portable,
                &bindings,
                &output,
                bytemuck::cast_slice(&push),
                groups,
                output_dim * input_dim,
                20,
            )?;
            let staged_ms = median_dispatch_ms(
                &device,
                &staged,
                &bindings,
                &output,
                bytemuck::cast_slice(&push),
                groups,
                output_dim * input_dim,
                20,
            )?;
            eprintln!(
                "RDNA3 weight grad M={rows} K={input_dim} N={output_dim}: portable {portable_ms:.4} ms, hardware {staged_ms:.4} ms, ratio {:.3}x",
                portable_ms / staged_ms
            );
        }
    }
    eprintln!(
        "RDNA3 hardware FMA weight grad: {words} words exact on {}",
        device.name()
    );
    Ok(())
}

fn ab_report(
    device: &VulkanDevice,
    label: &str,
    portable: &vulkan::ComputeKernel,
    staged: &vulkan::ComputeKernel,
    bindings: &[&GpuBuffer],
    output: &GpuBuffer,
    push: &[u8],
    groups: [u32; 3],
    len: usize,
) -> Result<()> {
    let reference = dispatch_words(device, portable, bindings, output, push, groups, len)?;
    let candidate = dispatch_words(device, staged, bindings, output, push, groups, len)?;
    assert_words_equal(label, &candidate, &reference)?;
    let portable_ms =
        median_dispatch_ms(device, portable, bindings, output, push, groups, len, 30)?;
    let staged_ms = median_dispatch_ms(device, staged, bindings, output, push, groups, len, 30)?;
    eprintln!(
        "AB {label}: portable {portable_ms:.4} ms, hardware {staged_ms:.4} ms, ratio {:.3}x",
        portable_ms / staged_ms
    );
    Ok(())
}

#[test]
fn rdna3_hw_fma_vision_ab_geometry_matrix() -> Result<()> {
    let device = VulkanDevice::new()?;
    let forward = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
            4,
            16,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_hw_fma_amd_rdna3.spv"),
            4,
            16,
        )?,
    );
    for (rows, input_dim, output_dim) in [
        (2usize, 16usize, 16usize),
        (6, 16, 16),
        (16, 16, 16),
        (32, 16, 16),
        (36, 16, 16),
        (36, 16, 32),
        (36, 32, 16),
        (8, 64, 64),
        (16, 16, 64),
    ] {
        let input = GpuBuffer::from_f32(&device, &deterministic_values(rows * input_dim, 5))?;
        let weight =
            GpuBuffer::from_f32(&device, &deterministic_values(output_dim * input_dim, 6))?;
        let bias = GpuBuffer::from_f32(&device, &deterministic_values(output_dim, 7))?;
        let output = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let push = [rows as u32, input_dim as u32, output_dim as u32, 1];
        let groups = [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1];
        ab_report(
            &device,
            &format!("forward M={rows} K={input_dim} N={output_dim}"),
            &forward.0,
            &forward.1,
            &[&input, &weight, &bias],
            &output,
            bytemuck::cast_slice(&push),
            groups,
            rows * output_dim,
        )?;
    }

    let input_gradient = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv"),
            4,
            16,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_input_grad_hw_fma_amd_rdna3.spv"),
            4,
            16,
        )?,
    );
    let dummy = GpuBuffer::from_f32(&device, &[0.0])?;
    for (rows, dy_dim, dx_dim) in [
        (2usize, 16usize, 16usize),
        (6, 16, 16),
        (16, 16, 16),
        (32, 16, 16),
        (36, 16, 16),
        (36, 32, 16),
        (8, 64, 64),
    ] {
        let gradient = GpuBuffer::from_f32(&device, &deterministic_values(rows * dy_dim, 8))?;
        let weight = GpuBuffer::from_f32(&device, &deterministic_values(dy_dim * dx_dim, 9))?;
        let output = GpuBuffer::zeros_f32(&device, rows * dx_dim)?;
        let push = [rows as u32, dy_dim as u32, dx_dim as u32, 0];
        let groups = [div_ceil_u32(dx_dim, 16), div_ceil_u32(rows, 16), 1];
        ab_report(
            &device,
            &format!("dX M={rows} K={dy_dim} N={dx_dim}"),
            &input_gradient.0,
            &input_gradient.1,
            &[&gradient, &weight, &dummy],
            &output,
            bytemuck::cast_slice(&push),
            groups,
            rows * dx_dim,
        )?;
    }

    let weight_gradient = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_weight_grad.spv"),
            3,
            12,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_weight_grad_hw_fma_amd_rdna3.spv"),
            3,
            12,
        )?,
    );
    for (rows, input_dim, output_dim) in [
        (4usize, 16usize, 16usize),
        (8, 12, 16),
        (16, 16, 16),
        (36, 16, 16),
        (36, 16, 32),
    ] {
        let input = GpuBuffer::from_f32(&device, &deterministic_values(rows * input_dim, 10))?;
        let gradient = GpuBuffer::from_f32(&device, &deterministic_values(rows * output_dim, 11))?;
        let output = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
        let push = [rows as u32, input_dim as u32, output_dim as u32];
        let groups = [div_ceil_u32(input_dim, 16), div_ceil_u32(output_dim, 16), 1];
        ab_report(
            &device,
            &format!("dW M={rows} K={input_dim} N={output_dim}"),
            &weight_gradient.0,
            &weight_gradient.1,
            &[&input, &gradient],
            &output,
            bytemuck::cast_slice(&push),
            groups,
            output_dim * input_dim,
        )?;
    }

    let patch = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_patch_linear_cpu_avx2.spv"),
            3,
            16,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_patch_linear_hw_fma_amd_rdna3.spv"),
            3,
            16,
        )?,
    );
    for (rows, input_dim, output_dim, channels) in [
        (32usize, 12usize, 16usize, 3usize),
        (36, 48, 16, 3),
        (36, 48, 32, 3),
    ] {
        let input = GpuBuffer::from_f32(&device, &deterministic_values(rows * input_dim, 12))?;
        let weight =
            GpuBuffer::from_f32(&device, &deterministic_values(output_dim * input_dim, 13))?;
        let output = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let push = PatchLinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            channels: channels as u32,
        };
        let groups = [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1];
        ab_report(
            &device,
            &format!("patch M={rows} K={input_dim} N={output_dim}"),
            &patch.0,
            &patch.1,
            &[&input, &weight],
            &output,
            bytemuck::bytes_of(&push),
            groups,
            rows * output_dim,
        )?;
    }
    Ok(())
}

#[test]
fn rdna3_hw_fma_operator_scale_ab() -> Result<()> {
    let device = VulkanDevice::new()?;
    let forward = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
            4,
            16,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_hw_fma_amd_rdna3.spv"),
            4,
            16,
        )?,
    );
    let (rows, input_dim, output_dim) = (512usize, 64usize, 64usize);
    let input = GpuBuffer::from_f32(&device, &deterministic_values(rows * input_dim, 21))?;
    let weight = GpuBuffer::from_f32(&device, &deterministic_values(output_dim * input_dim, 22))?;
    let bias = GpuBuffer::from_f32(&device, &deterministic_values(output_dim, 23))?;
    let output = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
    let push = [rows as u32, input_dim as u32, output_dim as u32, 1];
    let groups = [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1];
    ab_report(
        &device,
        "forward M=512 K=64 N=64",
        &forward.0,
        &forward.1,
        &[&input, &weight, &bias],
        &output,
        bytemuck::cast_slice(&push),
        groups,
        rows * output_dim,
    )?;

    let input_gradient = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv"),
            4,
            16,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_input_grad_hw_fma_amd_rdna3.spv"),
            4,
            16,
        )?,
    );
    let dummy = GpuBuffer::from_f32(&device, &[0.0])?;
    let output = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
    let push = [rows as u32, output_dim as u32, input_dim as u32, 0];
    let groups = [div_ceil_u32(input_dim, 16), div_ceil_u32(rows, 16), 1];
    ab_report(
        &device,
        "dX M=512 K=64 N=64",
        &input_gradient.0,
        &input_gradient.1,
        &[&input, &weight, &dummy],
        &output,
        bytemuck::cast_slice(&push),
        groups,
        rows * input_dim,
    )?;

    let weight_gradient = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_weight_grad.spv"),
            3,
            12,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_linear_weight_grad_hw_fma_amd_rdna3.spv"),
            3,
            12,
        )?,
    );
    let output = GpuBuffer::zeros_f32(&device, output_dim * input_dim)?;
    let push = [rows as u32, input_dim as u32, output_dim as u32];
    let groups = [div_ceil_u32(input_dim, 16), div_ceil_u32(output_dim, 16), 1];
    ab_report(
        &device,
        "dW M=512 K=64 N=64",
        &weight_gradient.0,
        &weight_gradient.1,
        &[&input, &weight],
        &output,
        bytemuck::cast_slice(&push),
        groups,
        output_dim * input_dim,
    )?;

    let patch = (
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_patch_linear_cpu_avx2.spv"),
            3,
            16,
        )?,
        vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_patch_linear_hw_fma_amd_rdna3.spv"),
            3,
            16,
        )?,
    );
    let (patch_rows, patch_input, patch_output) = (576usize, 48usize, 16usize);
    let patch_input_values =
        GpuBuffer::from_f32(&device, &deterministic_values(patch_rows * patch_input, 24))?;
    let patch_weight = GpuBuffer::from_f32(
        &device,
        &deterministic_values(patch_output * patch_input, 25),
    )?;
    let patch_output_values = GpuBuffer::zeros_f32(&device, patch_rows * patch_output)?;
    let patch_push = PatchLinearPush {
        rows: patch_rows as u32,
        input_dim: patch_input as u32,
        output_dim: patch_output as u32,
        channels: 3,
    };
    ab_report(
        &device,
        "patch M=576 K=48 N=16",
        &patch.0,
        &patch.1,
        &[&patch_input_values, &patch_weight],
        &patch_output_values,
        bytemuck::bytes_of(&patch_push),
        [
            div_ceil_u32(patch_output, 16),
            div_ceil_u32(patch_rows, 16),
            1,
        ],
        patch_rows * patch_output,
    )?;
    Ok(())
}

#[test]
fn rdna3_hw_fma_vision_gate_is_rdna3_only() {
    assert!(crate::vendor::rdna3_vision_capability(0x1002, 0x15bf));
    assert!(crate::vendor::rdna3_vision_capability(0x1002, 0x7448));
    for (vendor, device) in [(0x8086_u32, 0x1916_u32), (0x1002, 0x73ff), (0x10de, 0x2684)] {
        assert!(!crate::vendor::rdna3_vision_capability(vendor, device));
    }
}

fn spirv_workgroup_declaration(spv: &[u8]) -> (Vec<u32>, usize) {
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte word")))
        .collect();
    let mut offset = 5usize;
    let mut local_size = Vec::new();
    let mut workgroup_variables = 0usize;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let opcode = (words[offset] & 0xffff) as usize;
        assert!(
            count > 0 && offset + count <= words.len(),
            "malformed SPIR-V"
        );
        if opcode == 16 && count == 6 && words[offset + 2] == 17 {
            local_size = words[offset + 3..offset + 6].to_vec();
        }
        if opcode == 59 && count >= 4 && words[offset + 3] == 4 {
            workgroup_variables += 1;
        }
        offset += count;
    }
    (local_size, workgroup_variables)
}

#[test]
fn rdna3_hw_fma_linear_matches_the_software_construction() -> Result<()> {
    let device = VulkanDevice::new()?;
    let software = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
        4,
        16,
    )?;
    let hardware = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_hw_fma_amd_rdna3.spv"),
        4,
        16,
    )?;
    let corpus: Corpus = serde_json::from_slice(include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/fixture.json"
    ))?;
    let mut state = 0x1234_5678_u64;
    let mut values = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 40) as u32 as f32) / ((1u32 << 24) as f32);
                unit * 4.0 - 2.0
            })
            .collect()
    };
    let mut geometries: Vec<(usize, usize, usize)> = corpus
        .cases
        .iter()
        .map(|case| (case.rows, case.input_dim, case.output_dim))
        .collect();
    for rows in [2usize, 3, 4, 5, 6, 7, 8, 12, 16, 17, 32, 36] {
        for input_dim in [8usize, 12, 16, 32, 64] {
            for output_dim in [16usize, 32, 64] {
                geometries.push((rows, input_dim, output_dim));
            }
        }
    }
    let mut words = 0usize;
    for &(rows, input_dim, output_dim) in &geometries {
        let input = GpuBuffer::from_f32(&device, &values(rows * input_dim))?;
        let weight = GpuBuffer::from_f32(&device, &values(output_dim * input_dim))?;
        let bias = GpuBuffer::from_f32(&device, &values(output_dim))?;
        let output = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
        let push = [rows as u32, input_dim as u32, output_dim as u32, 1];
        let groups = [div_ceil_u32(output_dim, 16), div_ceil_u32(rows, 16), 1];
        let bindings = [&input, &weight, &bias];
        let reference = dispatch_words(
            &device,
            &software,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            rows * output_dim,
        )?;
        let candidate = dispatch_words(
            &device,
            &hardware,
            &bindings,
            &output,
            bytemuck::cast_slice(&push),
            groups,
            rows * output_dim,
        )?;
        assert_words_equal(
            &format!("RDNA3 hardware-FMA forward M={rows} K={input_dim} N={output_dim}"),
            &candidate,
            &reference,
        )?;
        words += reference.len();
    }
    eprintln!(
        "RDNA3 hardware FMA vs software FMA: {words} words exact across {} forward geometries on {}",
        geometries.len(),
        device.name()
    );
    Ok(())
}

#[test]
fn rdna3_hw_fma_vision_modules_declare_the_expected_workgroup() {
    for spv in [
        include_bytes!("../../shaders/vision_linear_hw_fma_amd_rdna3.spv").as_slice(),
        include_bytes!("../../shaders/vision_linear_input_grad_hw_fma_amd_rdna3.spv").as_slice(),
        include_bytes!("../../shaders/vision_patch_linear_hw_fma_amd_rdna3.spv").as_slice(),
        include_bytes!("../../shaders/vision_linear_weight_grad_hw_fma_amd_rdna3.spv").as_slice(),
    ] {
        assert_eq!(spirv_workgroup_declaration(spv), (vec![16, 16, 1], 0));
    }
}
