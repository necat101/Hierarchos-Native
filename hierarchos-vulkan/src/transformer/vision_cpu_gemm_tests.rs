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
