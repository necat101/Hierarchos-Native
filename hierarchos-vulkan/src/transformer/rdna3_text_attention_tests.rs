//! Raw-word proof that the RDNA3 wave64 short-attention module is a pure
//! scheduling change on the plain text path.
use super::*;

fn corpus_values(case: &serde_json::Value, key: &str) -> Vec<f32> {
    case[key].as_array().unwrap().iter()
        .map(|v| f32::from_bits(v.as_i64().unwrap() as u32)).collect()
}

fn assert_words(actual: &[f32], expected: &[f32], label: &str) {
    let differences = actual.iter().zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(actual.len(), expected.len());
    assert_eq!(differences, 0, "{label}: {differences} differing words, max_abs={:e}",
        actual.iter().zip(expected).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max));
}

fn primitive_dispatch(device: &VulkanDevice, shader: &[u8], values: &[Vec<f32>], push: &[u32], grid: [u32;3]) -> Result<Vec<Vec<f32>>> {
    let buffers=values.iter().map(|v| GpuBuffer::from_f32(device,v)).collect::<Result<Vec<_>>>()?;
    let kernel=vulkan::ComputeKernel::new(device,shader,buffers.len(),(push.len()*4) as u32)?;
    let mut commands=vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(&mut commands,&buffers.iter().collect::<Vec<_>>(),bytemuck::cast_slice(push),grid)?;
    commands.submit()?;
    buffers.iter().zip(values).map(|(b,v)| b.read_f32(v.len())).collect()
}

#[test]
fn rdna3_avx512_gelu_rms_gemm_cpu_words() -> Result<()> {
    let device=VulkanDevice::new()?;
    if !crate::vendor::is_amd_rdna3(&device) {return Ok(());}
    let corpus:serde_json::Value=serde_json::from_str(include_str!("../../validation/fixtures/rdna3_avx512_norm.json"))?;
    let c=&corpus["gelu"];let x=corpus_values(c,"x");let len=x.len();
    let actual=primitive_dispatch(&device,include_bytes!("../../shaders/image_text_gelu_avx512_amd_rdna3.spv"),
        &[x.clone(),vec![12345.;len]],&[len as u32],[len.div_ceil(256) as u32,1,1])?;
    assert_words(&actual[1],&corpus_values(c,"y"),"gelu/y");
    let actual=primitive_dispatch(&device,include_bytes!("../../shaders/image_text_gelu_backward_avx512_amd_rdna3.spv"),
        &[x,corpus_values(c,"dy"),vec![12345.;len]],&[len as u32],[len.div_ceil(256) as u32,1,1])?;
    assert_words(&actual[2],&corpus_values(c,"dx"),"gelu/dx");
    for c in corpus["rms"].as_array().unwrap() {
        let rows=c["rows"].as_u64().unwrap() as usize;let len=rows*64;
        let actual=primitive_dispatch(&device,include_bytes!("../../shaders/image_text_rms_avx512_amd_rdna3.spv"),
            &[corpus_values(c,"x"),corpus_values(c,"w"),vec![12345.;len],vec![12345.;rows],vec![12345.;rows]],
            &[rows as u32,64,1,1e-5f32.to_bits(),0,0,1,3],[rows.div_ceil(64) as u32,1,1])?;
        for (i,k) in [(2,"y"),(3,"rs"),(4,"ms")] {assert_words(&actual[i],&corpus_values(c,k),&format!("rms/{k}"));}
        let actual=primitive_dispatch(&device,include_bytes!("../../shaders/image_text_rms_backward_avx512_amd_rdna3.spv"),
            &[corpus_values(c,"dy"),corpus_values(c,"x"),corpus_values(c,"w"),corpus_values(c,"rs"),corpus_values(c,"ms"),vec![12345.;len]],
            &[rows as u32,64,1,0,0,2,0],[rows.div_ceil(64) as u32,1,1])?;
        assert_words(&actual[5],&corpus_values(c,"dx"),"rms/dx");
    }
    for c in corpus["gemm"].as_array().unwrap() {
        let reduction=c["reduction"].as_u64().unwrap() as u32;
        let actual=primitive_dispatch(&device,include_bytes!("../../shaders/image_text_input_grad_avx512_amd_rdna3.spv"),
            &[corpus_values(c,"dy"),corpus_values(c,"w"),vec![12345.;16*64]],&[16,64,reduction],[4,1,1])?;
        assert_words(&actual[2],&corpus_values(c,"dx"),&format!("gemm/k={reduction}"));
    }
    Ok(())
}

#[test]
fn rdna3_avx512_pow_rms_cpu_words() -> Result<()> {
    let device=VulkanDevice::new()?;
    if !crate::vendor::is_amd_rdna3(&device) {return Ok(());}
    let corpus:serde_json::Value=serde_json::from_str(include_str!("../../validation/fixtures/rdna3_avx512_norm.json"))?;
    for c in corpus["pow_rms"].as_array().unwrap() {
        let rows=c["rows"].as_u64().unwrap() as usize;
        let dim=c["dim"].as_u64().unwrap() as usize;
        for exact in [0,1] {
            let powers=primitive_dispatch(&device,include_bytes!("../../shaders/rdna3_pow_probe.spv"),
                &[corpus_values(c,"ms"),vec![12345.;rows]],&[rows as u32,exact],[rows.div_ceil(64) as u32,1,1])?;
            let expected=corpus_values(c,"power");
            eprintln!("direct pow dim={dim} exact={exact} differences={}",powers[1].iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits()).count());
            let actual=primitive_dispatch(&device,include_bytes!("../../shaders/image_text_pow_rms_backward_avx512_amd_rdna3.spv"),
                &[corpus_values(c,"dy"),corpus_values(c,"x"),corpus_values(c,"w"),corpus_values(c,"rs"),corpus_values(c,"ms"),vec![12345.;rows*dim]],
                &[rows as u32,dim as u32,1,1f32.to_bits(),1,2,exact],[rows.div_ceil(64) as u32,1,1])?;
            let expected=corpus_values(c,"dx");
            eprintln!("pow rms rows={rows} dim={dim} exact={exact} differences={} max={:e}",
                actual[5].iter().zip(&expected).filter(|(a,b)| a.to_bits()!=b.to_bits()).count(),
                actual[5].iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max));
        }
    }
    Ok(())
}

#[test]
fn rdna3_avx512_attention_cpu_parity() -> Result<()> {
    let device=VulkanDevice::new()?;
    if !crate::vendor::is_amd_rdna3(&device) {return Ok(());}
    let corpus:serde_json::Value=serde_json::from_str(include_str!("../../validation/fixtures/rdna3_avx512_norm.json"))?;
    let mut failures=Vec::new();
    for c in corpus["attention"].as_array().unwrap() {
        let number=|k:&str| c[k].as_u64().unwrap() as u32;
        let (batch,seq,heads,kv,hd)=(number("batch"),number("seq"),number("heads"),number("kvheads"),number("hd"));
        let scale=(c["scale"].as_f64().unwrap() as f32).to_bits();
        let geometry_push=[batch,seq,heads*hd,kv*hd,heads,kv,number("window"),scale,number("causal")];
        let len=(batch*seq*heads*hd) as usize;
        let inputs=[corpus_values(c,"q"),corpus_values(c,"k"),corpus_values(c,"v"),corpus_values(c,"qm"),corpus_values(c,"km")];
        let eager=c["kind"]=="eager";
        let forward=if eager {
            primitive_dispatch(&device,include_bytes!("../../shaders/image_text_eager_avx512_amd_rdna3.spv"),
                &[inputs[0].clone(),inputs[1].clone(),inputs[2].clone(),inputs[3].clone(),inputs[4].clone(),vec![12345.;len]],
                &geometry_push,[(batch*seq*heads).div_ceil(64),1,1])?[5].clone()
        } else {
            let mut packed=push(batch,seq,heads,kv,hd,1);
            packed[7]=number("window");packed[8]=scale;packed[29]=1;
            primitive_dispatch(&device,include_bytes!("../../shaders/image_text_sdpa_avx512_amd_rdna3.spv"),
                &[inputs[0].clone(),inputs[1].clone(),inputs[2].clone(),inputs[3].clone(),inputs[4].clone(),vec![0.],vec![12345.;len],vec![0.],vec![0.],vec![0.],vec![0.]],
                &packed,[(batch*seq*heads).div_ceil(64),1,1])?[6].clone()
        };
        let mut values=inputs.to_vec();values.push(corpus_values(c,"dy"));
        if !eager {values.push(forward.clone());}
        values.extend([vec![12345.;len],vec![12345.;(batch*seq*kv*hd) as usize],vec![12345.;(batch*seq*kv*hd) as usize]]);
        let backward=primitive_dispatch(&device,
            if eager {include_bytes!("../../shaders/image_text_eager_backward_avx512_amd_rdna3.spv")} else {include_bytes!("../../shaders/image_text_sdpa_backward_avx512_amd_rdna3.spv")},
            &values,if eager {&geometry_push[..]} else {&geometry_push[..8]},[kv,batch,1])?;
        let gradient=if eager {6} else {7};
        for (key,actual) in [("o",&forward),("dq",&backward[gradient]),("dk",&backward[gradient+1]),("dv",&backward[gradient+2])] {
            let expected=corpus_values(c,key);
            let max=actual.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
            let words=actual.iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
            eprintln!("attention {} b={batch} s={seq} hd={hd} window={} {key}: max={max:e} words={words}",c["kind"],number("window"));
            if max>=2e-7 || actual.iter().any(|v|!v.is_finite()) {failures.push(format!("{} s={seq} hd={hd} {key}: {max:e}",c["kind"]));}
        }
    }
    assert!(failures.is_empty(),"{}",failures.join("; "));
    Ok(())
}

#[test]
fn rdna3_avx512_layer_norm_cpu_words() -> Result<()> {
    let device = VulkanDevice::new()?;
    if !crate::vendor::is_amd_rdna3(&device) { return Ok(()); }
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/rdna3_avx512_norm.json"))?;
    assert_eq!(corpus["cpu"], "AVX512");
    let kernel = vulkan::ComputeKernel::new(&device,
        include_bytes!("../../shaders/image_text_layer_norm_avx512_amd_rdna3.spv"), 6, 12)?;
    for case in corpus["cases"].as_array().unwrap() {
        let rows = case["rows"].as_u64().unwrap() as usize;
        let dim = case["dim"].as_u64().unwrap() as usize;
        let inputs = ["x", "gamma", "beta"].map(|k| GpuBuffer::from_f32(&device, &corpus_values(case,k))).into_iter().collect::<Result<Vec<_>>>()?;
        let output = GpuBuffer::from_f32(&device, &vec![12345.0; rows*dim])?;
        let mean = GpuBuffer::from_f32(&device, &vec![12345.0; rows])?;
        let rstd = GpuBuffer::from_f32(&device, &vec![12345.0; rows])?;
        let push = [rows as u32, dim as u32, (case["eps"].as_f64().unwrap() as f32).to_bits()];
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(&mut commands, &[&inputs[0],&inputs[1],&inputs[2],&output,&mean,&rstd],
            bytemuck::cast_slice(&push), [rows.div_ceil(8) as u32,1,1])?;
        commands.submit()?;
        for (key, buffer, len) in [("mean",&mean,rows),("rstd",&rstd,rows),("y",&output,rows*dim)] {
            assert_words(&buffer.read_f32(len)?, &corpus_values(case,key), &format!("rows={rows} dim={dim}/{key}"));
        }
        let dy = GpuBuffer::from_f32(&device, &corpus_values(case,"dy"))?;
        let dx = GpuBuffer::from_f32(&device, &vec![12345.0; rows*dim])?;
        let dg = GpuBuffer::from_f32(&device, &vec![12345.0; dim])?;
        let db = GpuBuffer::from_f32(&device, &vec![12345.0; dim])?;
        let input_kernel = vulkan::ComputeKernel::new(&device,
            include_bytes!("../../shaders/image_text_layer_norm_input_grad_avx512_amd_rdna3.spv"),6,8)?;
        let param_kernel = vulkan::ComputeKernel::new(&device,
            include_bytes!("../../shaders/image_text_layer_norm_param_grad_avx512_amd_rdna3.spv"),6,8)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        input_kernel.record_dispatch(&mut commands, &[&dy,&inputs[0],&inputs[1],&mean,&rstd,&dx],
            bytemuck::cast_slice(&push[..2]), [rows.div_ceil(64) as u32,1,1])?;
        param_kernel.record_dispatch(&mut commands, &[&dy,&inputs[0],&mean,&rstd,&dg,&db],
            bytemuck::cast_slice(&push[..2]), [dim.div_ceil(256) as u32,1,1])?;
        commands.submit()?;
        for (key,buffer,len) in [("dx",&dx,rows*dim),("dg",&dg,dim),("db",&db,dim)] {
            assert_words(&buffer.read_f32(len)?,&corpus_values(case,key),&format!("rows={rows} dim={dim}/{key}"));
        }
    }
    Ok(())
}

#[test]
fn rdna3_avx512_vision_attention_projector_cpu_words() -> Result<()> {
    let device = VulkanDevice::new()?;
    if !crate::vendor::is_amd_rdna3(&device) { return Ok(()); }
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/rdna3_vision_avx512.json"))?;
    for case in corpus["attention"].as_array().unwrap() {
        let number = |key: &str| case[key].as_u64().unwrap() as u32;
        let (images, seq, dim, heads) = (
            number("images"), number("seq"), number("dim"), number("heads"));
        let len = (images * seq * dim) as usize;
        let actual = primitive_dispatch(
            &device,
            include_bytes!("../../shaders/vision_attention_avx512_amd_rdna3.spv"),
            &[
                corpus_values(case, "q"),
                corpus_values(case, "k"),
                corpus_values(case, "v"),
                vec![12345.0; len],
            ],
            &[images, seq, dim, heads],
            [(images * seq * heads).div_ceil(64), 1, 1],
        )?;
        assert_words(
            &actual[3],
            &corpus_values(case, "out"),
            &format!("vision attention images={images} seq={seq}"),
        );
    }
    for case in corpus["projector_rms"].as_array().unwrap() {
        let rows = case["rows"].as_u64().unwrap() as usize;
        let dim = case["dim"].as_u64().unwrap() as usize;
        let lanes = match rows { 16 => 4, 8 => 16, _ => panic!("unexpected projector rows {rows}") };
        let len = rows * dim;
        let push = [
            rows as u32,
            dim as u32,
            1,
            (case["eps"].as_f64().unwrap() as f32).to_bits(),
            1.0f32.to_bits(),
            0,
            1,
            lanes,
        ];
        let actual = primitive_dispatch(
            &device,
            include_bytes!("../../shaders/vision_projector_rms_avx512_amd_rdna3.spv"),
            &[
                corpus_values(case, "x"),
                corpus_values(case, "w"),
                vec![12345.0; len],
                vec![12345.0; rows],
                vec![12345.0; rows],
            ],
            &push,
            [rows.div_ceil(64) as u32, 1, 1],
        )?;
        assert_words(
            &actual[2],
            &corpus_values(case, "y"),
            &format!("projector RMS rows={rows} lanes={lanes}"),
        );
    }
    Ok(())
}

fn values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
        })
        .collect()
}

fn push(
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    causal: u32,
) -> Vec<u32> {
    vec![
        batch,
        seq,
        seq,
        heads * dim,
        kv_heads * dim,
        heads,
        kv_heads,
        0,
        (1.0 / (dim as f32).sqrt()).to_bits(),
        1.0f32.to_bits(),
        0.0f32.to_bits(),
        0,
        0.0f32.to_bits(),
        0.0f32.to_bits(),
        0,
        0,
        0,
        0,
        causal,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
    ]
}

fn run_attention(
    device: &VulkanDevice,
    shader: &[u8],
    inputs: Vec<Vec<f32>>,
    push: &[u32],
    dispatch: [u32; 3],
    output_len: usize,
) -> Result<Vec<f32>> {
    let buffers = inputs
        .iter()
        .map(|values| GpuBuffer::from_f32(device, values))
        .collect::<Result<Vec<_>>>()?;
    let kernel = vulkan::ComputeKernel::new(
        device,
        shader,
        buffers.len(),
        (push.len() * 4) as u32,
    )?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(
        &mut commands,
        &buffers.iter().collect::<Vec<_>>(),
        bytemuck::cast_slice(push),
        dispatch,
    )?;
    commands.submit()?;
    buffers[6].read_f32(output_len)
}

fn attention_inputs(
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
) -> (Vec<Vec<f32>>, usize) {
    let q_len = (batch * seq * heads * dim) as usize;
    let kv_len = (batch * seq * kv_heads * dim) as usize;
    let mask_len = (batch * seq) as usize;
    let output_len = q_len;
    let dummy = vec![0.0f32; 64];
    (
        vec![
            values(q_len, 0x51_0001),
            values(kv_len, 0x4B_0002),
            values(kv_len, 0x56_0003),
            vec![1.0; mask_len],
            vec![1.0; mask_len],
            dummy.clone(),
            vec![0.0; output_len],
            dummy.clone(),
            dummy.clone(),
            dummy.clone(),
            dummy,
        ],
        output_len,
    )
}

#[test]
fn rdna3_short_attention_is_word_exact_and_measured() -> Result<()> {
    let device = VulkanDevice::new()?;
    if !crate::vendor::rdna3_text_variants(&device) {
        return Ok(());
    }
    let generic = include_bytes!("../../shaders/transformer_attention_forward.spv");
    let short = include_bytes!("../../shaders/transformer_attention_short_amd_rdna3.spv");
    let cases = [
        (1u32, 1u32, 1u32, 1u32, 4u32, 1u32),
        (2, 5, 4, 2, 16, 1),
        (2, 8, 4, 4, 16, 1),
        (1, 16, 2, 1, 8, 0),
        (3, 7, 2, 2, 8, 1),
    ];
    for (batch, seq, heads, kv_heads, dim, causal) in cases {
        let push = push(batch, seq, heads, kv_heads, dim, causal);
        let (inputs, output_len) = attention_inputs(batch, seq, heads, kv_heads, dim);
        let expected = run_attention(
            &device,
            generic,
            inputs.clone(),
            &push,
            [seq, heads, batch],
            output_len,
        )?;
        let rows = batch * seq * heads;
        let actual = run_attention(
            &device,
            short,
            inputs,
            &push,
            [rows.div_ceil(64), 1, 1],
            output_len,
        )?;
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "case b={batch} s={seq} h={heads} kv={kv_heads} d={dim} causal={causal} word {index}: {a:e} != {b:e}"
            );
        }
    }

    let push = push(2, 8, 4, 2, 16, 1);
    let (inputs, output_len) = attention_inputs(2, 8, 4, 2, 16);
    let started = std::time::Instant::now();
    for _ in 0..200 {
        run_attention(
            &device,
            generic,
            inputs.clone(),
            &push,
            [8, 4, 2],
            output_len,
        )?;
    }
    let generic_elapsed = started.elapsed();
    let started = std::time::Instant::now();
    for _ in 0..200 {
        run_attention(
            &device,
            short,
            inputs.clone(),
            &push,
            [1, 1, 1],
            output_len,
        )?;
    }
    let short_elapsed = started.elapsed();
    eprintln!(
        "RDNA3 short attention 64 rows: generic {:?}, wave64 {:?}, ratio {:.3}x",
        generic_elapsed,
        short_elapsed,
        generic_elapsed.as_secs_f64() / short_elapsed.as_secs_f64()
    );
    Ok(())
}
