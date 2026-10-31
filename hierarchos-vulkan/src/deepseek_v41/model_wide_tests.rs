use super::*;

fn generated_values(mut state: u32, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f32::from_bits(0x3d000000 | (state & 0x01ffffff) | ((state & 0x02000000) << 6))
        })
        .collect()
}

#[test]
fn deepseek_v41_wide_projection_panels_and_tied_topk_match_cpu() -> Result<()> {
    #[derive(Deserialize)]
    struct LinearCase {
        rows: usize,
        inner: usize,
        outputs: usize,
        seed: u32,
        input_sha256: String,
        weight_sha256: String,
        expected_bits: Vec<u32>,
        expected_bf16_bits: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct TopkCase {
        rows: usize,
        seq: usize,
        ratio: usize,
        start_pos: usize,
        offset: usize,
        width: usize,
        topk: usize,
        score_bits: Vec<u32>,
        expected_indices: Vec<i32>,
    }
    #[derive(Deserialize)]
    struct Kernels {
        linears: Vec<LinearCase>,
        topks: Vec<TopkCase>,
    }
    let f: Kernels = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_wide_kernels.json"
    ))?;
    assert_eq!(f.linears.len(), 24);
    assert_eq!(f.topks.len(), 68);
    let device = VulkanDevice::new()?;
    let linear = crate::deepseek_v41::indexer::LinearKernel::new(&device)?;
    let mixed = crate::deepseek_v41::DeepseekV41MixedLinearKernel::new(device.clone())?;
    for c in f.linears {
        let x = generated_values(c.seed, c.rows * c.inner);
        let w = generated_values(c.seed + 1, c.outputs * c.inner);
        assert_eq!(
            format!("{:x}", Sha256::digest(bytemuck::cast_slice::<f32, u8>(&x))),
            c.input_sha256
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(bytemuck::cast_slice::<f32, u8>(&w))),
            c.weight_sha256
        );
        let out = linear.apply(
            &device,
            &GpuBuffer::from_f32(&device, &x)?,
            &GpuBuffer::from_f32(&device, &w)?,
            c.rows,
            c.inner,
            c.outputs,
        )?;
        assert_eq!(
            bits(&out.read_f32(c.rows * c.outputs)?),
            c.expected_bits,
            "m{} k{} n{}",
            c.rows,
            c.inner,
            c.outputs
        );
        let rounded: Vec<_> = w
            .iter()
            .map(|v| {
                let b = v.to_bits();
                f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000)
            })
            .collect();
        let weight = crate::deepseek_v41::DeepseekV41LinearWeights::Bf16 {
            rows: c.outputs,
            cols: c.inner,
            values: rounded,
            lora: None,
        }
        .upload(&device)?;
        let out = mixed.apply(
            &GpuBuffer::from_f32(&device, &x)?,
            c.rows,
            &weight,
            DeepseekV41WeightOutput::Bf16,
        )?;
        assert_eq!(
            bits(&out.read_f32(c.rows * c.outputs)?),
            c.expected_bf16_bits,
            "BF16 m{} k{} n{}",
            c.rows,
            c.inner,
            c.outputs
        );
    }
    let topk = crate::deepseek_v41::DeepseekV41IndexTopk::new(device.clone())?;
    for c in f.topks {
        let scores = GpuBuffer::from_u32(&device, &c.score_bits)?;
        let out = topk.rank(
            &scores,
            c.rows,
            c.width,
            c.seq,
            c.start_pos,
            c.ratio,
            c.topk,
            c.offset,
        )?;
        let actual: Vec<_> = out
            .read_u32(c.rows * c.topk)?
            .iter()
            .map(|&x| x as i32)
            .collect();
        assert_eq!(
            actual, c.expected_indices,
            "tied top-k width {} k {}",
            c.width, c.topk
        );
    }
    Ok(())
}

#[derive(Deserialize)]
struct Checkpoint {
    directory: String,
    files: BTreeMap<String, String>,
    tensor_count: usize,
    packed_count: usize,
}

#[derive(Deserialize)]
struct WideCase {
    #[serde(flatten)]
    model: Fixture,
    checkpoint: Checkpoint,
}

#[test]
fn deepseek_v41_model_wider_yarn_bf16_strict_all_logits() -> Result<()> {
    const SOURCE: &str = include_str!("../../validation/fixtures/deepseek_v41_wide_model.json");
    #[derive(Deserialize)]
    struct Cases {
        cases: Vec<WideCase>,
    }
    let fixture: Cases = serde_json::from_str(SOURCE)?;
    assert_eq!(fixture.cases.len(), 3);
    let binding = format!("{:x}", Sha256::digest(SOURCE.as_bytes()));
    let device = VulkanDevice::new()?;
    let mut failures = vec![];
    let mut exported = vec![];
    let trace = std::env::var_os("HIERARCHOS_V41_WIDE_TRACE").is_some();
    for (case, (seed, hidden, head, index_heads, vocab, steps, prefill)) in
        fixture.cases.into_iter().zip([
            (4141071, 128, 128, 4, 97, 7, 65),
            (4141072, 256, 512, 32, 129, 7, 65),
            (4141073, 128, 128, 4, 97, 8, 512),
        ])
    {
        let f = case.model;
        assert_eq!(f.seed, seed);
        assert_eq!(f.steps.len(), steps);
        assert_eq!(f.stream_precision, "bf16");
        assert_eq!(f.revision, crate::deepseek_v41::REFERENCE_REVISION);
        assert_eq!(
            f.source_sha256,
            "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
        );
        assert_eq!(f.config["text_config"]["hidden_size"], hidden);
        assert_eq!(f.config["text_config"]["head_dim"], head);
        assert_eq!(f.config["text_config"]["index_n_heads"], index_heads);
        assert_eq!(f.config["text_config"]["vocab_size"], vocab);
        assert_eq!(f.steps[0].seq, prefill);
        assert_eq!(f.max_prefill, prefill);
        assert_eq!(case.checkpoint.packed_count, 103);
        assert_eq!(
            case.checkpoint.directory,
            format!("deepseek_v41_wide_{seed}")
        );
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("validation/fixtures")
            .join(&case.checkpoint.directory);
        for (name, expected) in &case.checkpoint.files {
            assert_eq!(std::path::Path::new(name).components().count(), 1);
            assert_eq!(
                format!("{:x}", Sha256::digest(fs::read(root.join(name))?)),
                *expected,
                "{name}"
            );
        }
        // Open the actual frozen on-disk package, including cross-shard scales.
        let mut checkpoint = DeepseekV41Checkpoint::open(&root)?;
        assert!(case.checkpoint.tensor_count > 65);
        let mut model = DeepseekV41CausalLM::from_checkpoint_with_precision(
            device.clone(),
            &mut checkpoint,
            runtime(&f),
            Some(f.tokens.clone()),
            DeepseekV41WeightOutput::Bf16,
        )?;
        assert_eq!(model.stream_precision(), DeepseekV41WeightOutput::Bf16);
        if trace {
            model.backbone.trace = Some(vec![]);
        }
        let mut steps = vec![];
        for step in &f.steps {
            let logits = run(&mut model, step)?;
            let shape = vec![logits.batch_size, logits.seq_len, logits.vocab_size];
            assert_eq!(shape, step.shape);
            let values = logits.read_all()?;
            assert!(!values.is_empty());
            assert_eq!(values.len(), step.logits.len());
            let mut worst = 0.0f64;
            for (&a, &b) in values.iter().zip(&step.logits) {
                assert!(a.is_finite() && b.is_finite());
                worst = worst.max((a as f64 - b as f64).abs());
            }
            assert!(values.iter().any(|&v| v != 0.0));
            eprintln!(
                "V4.1 wider BF16 seed {seed}, {}: {} logits, drift {worst:.17e}",
                step.name,
                values.len()
            );
            if worst >= 2e-7 {
                failures.push(format!("{seed}, {}: {worst:.17e}", step.name));
            }
            assert_eq!(model.next_position(), Some(step.pos + step.seq));
            let count = f.batch_size * step.seq * hidden;
            let (collapsed, normalized) = model.final_trace.as_ref().unwrap();
            let mut output = json!({"name":step.name, "shape":shape, "logits":values,
                "collapsed":collapsed.read_f32(count)?, "normalized":normalized.read_f32(count)?});
            if let Some(stages) = &model.backbone.trace {
                output["stages"] = json!(stages
                    .iter()
                    .map(|(layer, stage, buffer, len)| -> Result<_> {
                        Ok(json!({"layer":layer,"stage":stage,"values":buffer.read_f32(*len)?}))
                    })
                    .collect::<Result<Vec<_>>>()?);
                let mut selections = serde_json::Map::new();
                let mut caches = serde_json::Map::new();
                for layer in 0..6 {
                    if let Some(s) = model.backbone.index_selection(layer)? {
                        if s.width > 0 {
                            selections.insert(
                                layer.to_string(),
                                json!({"indices":s.read_indices()?,
                                "score_bits":s.scores.read_u32(s.rows*s.width)?}),
                            );
                        }
                    }
                    if let Some((kv, keys)) = model.backbone.attention_cache_trace(layer)? {
                        caches.insert(layer.to_string(), json!({"kv":kv,"keys":keys}));
                    }
                }
                output["index_diagnostics"] = json!(selections);
                output["cache_diagnostics"] = json!(caches);
            }
            steps.push(output);
        }
        exported.push(
            json!({"seed":seed,"revision":f.revision,"fixture_sha256":binding,
            "checkpoint_files":case.checkpoint.files,"steps":steps}),
        );
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/wide-model-native-logits.json");
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, serde_json::to_vec(&json!({"cases":exported}))?)?;
    assert!(
        failures.is_empty(),
        "strict <2e-7, rtol=0 failures: {failures:?}"
    );
    Ok(())
}
