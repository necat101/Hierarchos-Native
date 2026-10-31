use super::*;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

#[path = "model_streamed_tests.rs"]
mod streamed;
#[path = "model_structure_tests.rs"]
mod structure;
#[path = "model_vocabulary_tests.rs"]
mod vocabulary;
#[path = "model_wide_tests.rs"]
mod wide;
#[path = "model_training_tests.rs"]
mod training;

#[derive(Deserialize)]
struct Fixture {
    seed: u64,
    revision: String,
    source_sha256: String,
    config: Value,
    batch_size: usize,
    max_prefill: usize,
    max_positions: usize,
    tokens: DeepseekV41EngramTokens,
    tensors: BTreeMap<String, Tensor>,
    #[serde(default)]
    packed_tensors: BTreeMap<String, PackedTensor>,
    #[serde(default)]
    stream_precision: String,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct PackedTensor {
    rows: usize,
    cols: usize,
    dtype: String,
    data: Vec<u8>,
    scales: Vec<u8>,
}
#[derive(Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    values: Vec<f32>,
    #[serde(default)]
    dtype: String,
}
#[derive(Deserialize)]
struct Step {
    name: String,
    pos: usize,
    seq: usize,
    ids: Vec<u32>,
    mask: Vec<u32>,
    shape: Vec<usize>,
    logits: Vec<f32>,
}

fn fixture() -> Result<Fixture> {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_model.json"
    ))?;
    assert_eq!(f.seed, 4141017);
    assert_eq!(f.steps.len(), 6);
    assert_eq!(f.revision, crate::deepseek_v41::REFERENCE_REVISION);
    assert_eq!(
        f.source_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    Ok(f)
}
fn runtime(f: &Fixture) -> DeepseekV41AttentionRuntimeConfig {
    DeepseekV41AttentionRuntimeConfig {
        batch_size: f.batch_size,
        max_prefill: f.max_prefill,
        max_positions: f.max_positions,
    }
}

struct Package {
    root: PathBuf,
}
impl Package {
    fn new(f: &Fixture, sharded_bf16: bool) -> Result<Self> {
        static ID: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hierarchos-v41-lm-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        let result = Self { root };
        fs::write(
            result.root.join("config.json"),
            serde_json::to_vec(&f.config)?,
        )?;
        let mut data: Vec<_> = f
            .tensors
            .iter()
            .map(|(name, tensor)| {
                let bf = tensor.dtype == "bf16"
                    || sharded_bf16
                        && matches!(
                            name.as_str(),
                            "embed.weight" | "norm.weight" | "head.weight"
                        );
                let bytes = if bf {
                    tensor
                        .values
                        .iter()
                        .flat_map(|v| {
                            assert_eq!(v.to_bits() & 0xffff, 0, "BF16 fixture value");
                            ((v.to_bits() >> 16) as u16).to_le_bytes()
                        })
                        .collect::<Vec<_>>()
                } else {
                    bytemuck::cast_slice(&tensor.values).to_vec()
                };
                (
                    name.clone(),
                    tensor.shape.clone(),
                    if bf { Dtype::BF16 } else { Dtype::F32 },
                    bytes,
                )
            })
            .collect();
        for (name, tensor) in &f.packed_tensors {
            let fp4 = tensor.dtype == "fp4_row32";
            let row_scaled = tensor.dtype == "engram_fp8_row32";
            assert!(fp4 || row_scaled || tensor.dtype == "fp8_block32");
            let dtype = if fp4 { Dtype::F4 } else { Dtype::F8_E4M3 };
            data.push((
                name.clone(),
                vec![tensor.rows, tensor.cols],
                dtype,
                tensor.data.clone(),
            ));
            let scale_name = format!("{}.scale", name.strip_suffix(".weight").unwrap());
            let scale_rows = if fp4 || row_scaled {
                tensor.rows
            } else {
                tensor.rows.div_ceil(32)
            };
            data.push((
                scale_name,
                vec![scale_rows, tensor.cols / 32],
                Dtype::F8_E8M0,
                tensor.scales.clone(),
            ));
        }
        let mut map = BTreeMap::new();
        for shard in 0..if sharded_bf16 { 2 } else { 1 } {
            let filename = if sharded_bf16 {
                format!("weights-{shard}.safetensors")
            } else {
                "model.safetensors".to_owned()
            };
            let views = data
                .iter()
                .enumerate()
                .filter(|(i, _)| !sharded_bf16 || i % 2 == shard)
                .map(|(_, (name, shape, dtype, bytes))| {
                    map.insert(name.as_str(), filename.clone());
                    Ok((
                        name.as_str(),
                        TensorView::new(*dtype, shape.clone(), bytes)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            serialize_to_file(views, None, &result.root.join(filename))?;
        }
        if sharded_bf16 {
            fs::write(
                result.root.join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"weight_map":map}))?,
            )?;
        }
        Ok(result)
    }
    fn reader(&self) -> Result<DeepseekV41Checkpoint> {
        DeepseekV41Checkpoint::open(&self.root)
    }
}
impl Drop for Package {
    fn drop(&mut self) {
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let _ = fs::remove_dir(&self.root);
    }
}

fn load(device: &VulkanDevice, f: &Fixture, package: &Package) -> Result<DeepseekV41CausalLM> {
    if f.stream_precision == "bf16" {
        return DeepseekV41CausalLM::from_checkpoint_with_precision(
            device.clone(),
            &mut package.reader()?,
            runtime(f),
            Some(f.tokens.clone()),
            DeepseekV41WeightOutput::Bf16,
        );
    }
    assert!(f.stream_precision.is_empty());
    DeepseekV41CausalLM::from_checkpoint(
        device.clone(),
        &mut package.reader()?,
        runtime(f),
        Some(f.tokens.clone()),
    )
}
fn run(model: &mut DeepseekV41CausalLM, step: &Step) -> Result<DeepseekV41Logits> {
    if step.pos == 0 {
        model.prefill(&step.ids, step.seq, Some(&step.mask))
    } else {
        model.decode(&step.ids, step.pos, Some(&step.mask))
    }
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn deepseek_v41_model_exports_complete_fp32_diagnostic_logits() -> Result<()> {
    let f = fixture()?;
    let device = VulkanDevice::new()?;
    let package = Package::new(&f, false)?;
    let mut model = load(&device, &f, &package)?;
    model.backbone.trace = Some(vec![]);
    let mut output = vec![];
    for step in &f.steps {
        let logits = run(&mut model, step)?;
        let shape = vec![logits.batch_size, logits.seq_len, logits.vocab_size];
        assert_eq!(shape, step.shape);
        let values = logits.read_all()?;
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|&v| v != 0.0));
        assert_eq!(values.len(), step.logits.len());
        let drift = values
            .iter()
            .zip(&step.logits)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "V4.1 full-model FP32 diagnostic {}: {} logits, max drift {drift:.17e}",
            step.name,
            values.len()
        );
        assert_eq!(model.next_position(), Some(step.pos + step.seq));
        let trace = model.final_trace.as_ref().unwrap();
        let count = f.batch_size * step.seq * 32;
        let stages = model
            .backbone
            .trace
            .as_ref()
            .unwrap()
            .iter()
            .map(|(layer, stage, buffer, len)| {
                Ok(json!({"layer":layer,"stage":stage,"values":buffer.read_f32(*len)?}))
            })
            .collect::<Result<Vec<_>>>()?;
        output.push(json!({"name":step.name,"shape":shape,"logits":values,"collapsed":trace.0.read_f32(count)?,"normalized":trace.1.read_f32(count)?,"stages":stages}));
    }
    let destination = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/model-native-logits.json");
    fs::create_dir_all(destination.parent().unwrap())?;
    fs::write(
        destination,
        serde_json::to_vec(
            &json!({"revision":f.revision,"fixture_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../validation/fixtures/deepseek_v41_model.json"))),"steps":output}),
        )?,
    )?;
    // Export/shape/finite-value coverage is not an acceptance assertion. The
    // explicit strict test and Python comparator below remain the hard gate.
    Ok(())
}

#[test]
fn deepseek_v41_model_strict_all_logits() -> Result<()> {
    let f = fixture()?;
    let device = VulkanDevice::new()?;
    let package = Package::new(&f, false)?;
    let mut model = load(&device, &f, &package)?;
    for step in &f.steps {
        let output = run(&mut model, step)?;
        assert_eq!(
            vec![output.batch_size, output.seq_len, output.vocab_size],
            step.shape
        );
        let values = output.read_all()?;
        assert!(!values.is_empty());
        assert_eq!(values.len(), step.logits.len());
        for (index, (&actual, &expected)) in values.iter().zip(&step.logits).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            let drift = (actual as f64 - expected as f64).abs();
            assert!(
                drift < 2.0e-7,
                "{} logit {index}: {drift:.17e} must be <2e-7, rtol=0",
                step.name
            );
        }
    }
    Ok(())
}

/// The normal package path selects precision and every residency policy from
/// the checkpoint itself. The fixture supplies Engram's non-persistent
/// metadata inline, as a converted package with sidecar JSON would.
#[test]
fn deepseek_v41_default_package_loader_strict_all_logits() -> Result<()> {
    let source: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_bf16_model.json"
    ))?;
    let f: Fixture = serde_json::from_value(source["cases"][0].clone())?;
    assert_eq!(f.seed, 4141061);
    let package = Package::new(&f, false)?;
    // The released package has no tokenizer files; a precomputed sidecar keeps
    // the default loader free of both flags and external Python artifacts.
    fs::write(
        package.root.join("engram_tokens.json"),
        serde_json::to_vec(&f.tokens)?,
    )?;
    let device = VulkanDevice::new()?;
    let mut model =
        DeepseekV41CausalLM::from_hf_package(device, &package.root, f.batch_size, f.max_positions)?;
    assert_eq!(model.stream_precision(), DeepseekV41WeightOutput::Bf16);
    assert_eq!(model.resident_routed_expert_count(), 0);
    assert_eq!(model.resident_shared_expert_count(), 0);
    assert_eq!(model.resident_engram_table_count(), 0);
    assert_eq!(model.resident_projection_layer_count(), 0);
    assert_eq!(model.resident_vocabulary_tensor_count(), 0);
    for step in &f.steps {
        let output = run(&mut model, step)?;
        assert_eq!(
            vec![output.batch_size, output.seq_len, output.vocab_size],
            step.shape
        );
        let values = output.read_all()?;
        assert_eq!(values.len(), step.logits.len());
        let worst = values
            .iter()
            .zip(&step.logits)
            .map(|(&a, &b)| {
                assert!(a.is_finite() && b.is_finite());
                (a as f64 - b as f64).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(worst < 2e-7, "{}: {worst:.17e}", step.name);
        assert_eq!(model.next_position(), Some(step.pos + step.seq));
    }
    Ok(())
}

/// The default loader must not guess Engram metadata when a package has neither
/// tokenizer files nor precomputed artifacts.
#[test]
fn deepseek_v41_application_streaming_is_explicit_and_bit_exact() -> Result<()> {
    let source:Value=serde_json::from_str(include_str!("../../validation/fixtures/deepseek_v41_bf16_model.json"))?;
    for f in [fixture()?,serde_json::from_value(source["cases"][0].clone())?] {
        let package=Package::new(&f,false)?;
        fs::write(package.root.join("engram_tokens.json"),serde_json::to_vec(&f.tokens)?)?;
        let device=VulkanDevice::new()?;
        let mut resident=DeepseekV41CausalLM::from_hf_package_with_lora_and_streaming(device.clone(),&package.root,f.batch_size,f.max_positions,None,false)?;
        let mut streamed=DeepseekV41CausalLM::from_hf_package_with_lora_and_streaming(device.clone(),&package.root,f.batch_size,f.max_positions,None,true)?;
        assert!(resident.resident_routed_expert_count()>0);
        assert!(resident.resident_vocabulary_tensor_count()>0);
        assert_eq!(streamed.resident_routed_expert_count(),0);
        assert_eq!(streamed.resident_vocabulary_tensor_count(),0);
        assert_eq!(streamed.resident_shared_expert_count(),0);
        assert_eq!(streamed.resident_engram_table_count(),0);
        assert_eq!(streamed.resident_projection_layer_count(),0);
        for step in &f.steps {
            let a=run(&mut resident,step)?.read_all()?;
            let b=run(&mut streamed,step)?.read_all()?;
            assert_eq!(a.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let worst=b.iter().zip(&step.logits).map(|(&a,&b)|(a as f64-b as f64).abs()).fold(0.0f64,f64::max);
            assert!(worst<2e-7,"seed={} {} drift={worst}",f.seed,step.name);
        }
        eprintln!("V4.1 explicit application residency seed {}: CPU strict logits and resident/streamed bits PASS",f.seed);
    }
    Ok(())
}

#[test]
fn deepseek_v41_default_loader_requires_engram_metadata() -> Result<()> {
    let source: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_bf16_model.json"
    ))?;
    let f: Fixture = serde_json::from_value(source["cases"][0].clone())?;
    let package = Package::new(&f, false)?;
    let error = DeepseekV41CausalLM::from_hf_package(
        VulkanDevice::new()?,
        &package.root,
        f.batch_size,
        f.max_positions,
    )
    .err()
    .unwrap();
    let message = format!("{error:#}");
    assert!(
        message.contains("tokenizer.json or engram_tokens.json"),
        "{message}"
    );
    Ok(())
}

/// `prefill_last` must match a complete prefill, and the generation loop must
/// reproduce manual prefill/decode argmax steps on batch one.
#[test]
fn deepseek_v41_package_generation_matches_manual_steps() -> Result<()> {
    let source: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_bf16_model.json"
    ))?;
    let mut f: Fixture = serde_json::from_value(source["cases"][0].clone())?;
    f.config["text_config"]["engram_token_map"] = json!(f.tokens.token_map);
    f.config["text_config"]["engram_hash_multipliers"] = json!(f.tokens.multipliers);
    let package = Package::new(&f, false)?;
    let device = VulkanDevice::new()?;
    let mut model =
        DeepseekV41CausalLM::from_hf_package(device, &package.root, 1, f.max_positions)?;
    let prompt = f.steps[0].ids[..f.steps[0].seq].to_vec();
    let last = model.prefill_last(&prompt)?;
    let last_values = last.read_all()?;
    assert_eq!(last_values.len(), f.steps[0].shape[2]);
    assert_eq!(model.next_position(), Some(prompt.len()));
    let full = model.prefill(&prompt, prompt.len(), None)?.read_all()?;
    // The projection intentionally switches from the row GEMM to the vector
    // dot below four rows, so the last row may differ by one ulp while staying
    // inside the unchanged full-logit gate.
    for (&a, &b) in last_values
        .iter()
        .zip(&full[full.len() - f.steps[0].shape[2]..])
    {
        assert!((a as f64 - b as f64).abs() < 2e-7);
    }

    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let first = argmax(&model.prefill_last(&prompt)?.read_all()?);
    let second = argmax(&model.decode(&[first], prompt.len(), None)?.read_all()?);
    let config = crate::VulkanTransformerGenerationConfig {
        max_new_tokens: Some(2),
        max_length: None,
        do_sample: false,
        use_cache: true,
        ..Default::default()
    };
    let generated = model.generate_sequences_ids(&prompt, &config)?;
    assert_eq!(generated, vec![vec![first, second]]);
    Ok(())
}

#[test]
fn deepseek_v41_model_additional_seeds_and_top3_strict_all_logits() -> Result<()> {
    #[derive(Deserialize)]
    struct Additional {
        cases: Vec<Fixture>,
    }
    const SOURCE: &str =
        include_str!("../../validation/fixtures/deepseek_v41_model_additional.json");
    let f: Additional = serde_json::from_str(SOURCE)?;
    assert_eq!(f.cases.len(), 2);
    let binding = format!("{:x}", Sha256::digest(SOURCE.as_bytes()));
    let device = VulkanDevice::new()?;
    let mut exported = vec![];
    let mut failures = vec![];
    for (case, (seed, topk)) in f.cases.into_iter().zip([(4141018, 2), (4141019, 3)]) {
        assert_eq!(case.seed, seed);
        assert_eq!(case.steps.len(), 6);
        assert_eq!(case.config["text_config"]["num_experts_per_tok"], topk);
        assert_eq!(case.revision, crate::deepseek_v41::REFERENCE_REVISION);
        assert_eq!(
            case.source_sha256,
            "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
        );
        let package = Package::new(&case, false)?;
        let mut model = load(&device, &case, &package)?;
        let mut output = vec![];
        for step in &case.steps {
            let result = run(&mut model, step)?;
            let shape = vec![result.batch_size, result.seq_len, result.vocab_size];
            assert_eq!(shape, step.shape);
            let values = result.read_all()?;
            assert!(!values.is_empty());
            assert_eq!(values.len(), step.logits.len());
            let mut worst = 0.0f64;
            for (&actual, &expected) in values.iter().zip(&step.logits) {
                assert!(actual.is_finite() && expected.is_finite());
                worst = worst.max((actual as f64 - expected as f64).abs());
            }
            eprintln!(
                "V4.1 strict seed {}, {}: {worst:.17e}",
                case.seed, step.name
            );
            if worst >= 2e-7 {
                failures.push(format!("seed {}, {}: {worst:.17e}", case.seed, step.name));
            }
            output.push(json!({"name":step.name,"shape":shape,"logits":values}));
        }
        exported.push(json!({"seed":case.seed,"revision":case.revision,"fixture_sha256":binding,"steps":output}));
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/model-additional-native-logits.json");
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, serde_json::to_vec(&json!({"cases":exported}))?)?;
    assert!(
        failures.is_empty(),
        "strict <2e-7, rtol=0 failures: {failures:?}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_model_single_sharded_bf16_and_raw_weights_are_identical() -> Result<()> {
    let f = fixture()?;
    let device = VulkanDevice::new()?;
    let single = Package::new(&f, false)?;
    let sharded = Package::new(&f, true)?;
    let mut a = load(&device, &f, &single)?;
    let mut b = load(&device, &f, &sharded)?;
    let mut reader = single.reader()?;
    let c = reader.config.clone();
    let engram = DeepseekV41EngramRuntime::from_checkpoint(
        device.clone(),
        &mut reader,
        f.tokens.clone(),
        f.batch_size,
        f.max_positions,
    )?;
    let weights = DeepseekV41ModelWeights {
        embedding: reader.read_tensor("embed.weight", &[16, 32])?,
        norm: reader.read_tensor("norm.weight", &[32])?,
        head: reader.read_tensor("head.weight", &[16, 32])?,
        layers: (0..6)
            .map(|id| reader.load_backbone_layer(id))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut raw = DeepseekV41CausalLM::new(device, &c, runtime(&f), weights, Some(engram))?;
    for step in &f.steps {
        let expected = bits(&run(&mut a, step)?.read_all()?);
        assert_eq!(
            expected,
            bits(&run(&mut b, step)?.read_all()?),
            "sharded/BF16 {}",
            step.name
        );
        assert_eq!(
            expected,
            bits(&run(&mut raw, step)?.read_all()?),
            "raw constructor {}",
            step.name
        );
    }
    Ok(())
}

#[test]
fn deepseek_v41_model_packed_experts_strict_all_logits() -> Result<()> {
    packed_model_cases(
        include_str!("../../validation/fixtures/deepseek_v41_mixed_model.json"),
        [4141041, 4141042],
        72,
        "mixed-model-native-logits.json",
        false,
    )
}

#[test]
fn deepseek_v41_legacy_backbone_weights_still_promote_dense_bf16() -> Result<()> {
    let mut f = fixture()?;
    let name = "layers.1.attn.wo_a.weight";
    let tensor = f.tensors.get_mut(name).unwrap();
    tensor.dtype = "bf16".to_owned();
    for value in &mut tensor.values {
        let b = value.to_bits();
        *value = f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000);
    }
    let package = Package::new(&f, false)?;
    let mut checkpoint = package.reader()?;
    assert_eq!(checkpoint.tensor_dtype(name)?, "BF16");
    let layer = checkpoint.load_backbone_layer(1)?;
    assert_eq!(layer.attention.attention.wo_a, f.tensors[name].values);
    Ok(())
}

#[test]
fn deepseek_v41_model_packed_attention_indexer_strict_all_logits() -> Result<()> {
    packed_model_cases(
        include_str!("../../validation/fixtures/deepseek_v41_attention_model.json"),
        [4141051, 4141052],
        99,
        "attention-model-native-logits.json",
        true,
    )
}

#[test]
fn deepseek_v41_model_bf16_residual_hc_strict_all_logits() -> Result<()> {
    packed_model_cases(
        include_str!("../../validation/fixtures/deepseek_v41_bf16_model.json"),
        [4141061, 4141062],
        103,
        "bf16-model-native-logits.json",
        true,
    )
}

#[test]
fn deepseek_v41_streamed_hybrid_and_bf16_strict_all_logits() -> Result<()> {
    streamed_model_cases(
        crate::deepseek_v41::DeepseekV41ExpertResidency::Streamed,
        crate::deepseek_v41::DeepseekV41EngramResidency::Resident,
        crate::deepseek_v41::DeepseekV41ProjectionResidency::Resident,
        "streamed-",
    )
}

#[test]
fn deepseek_v41_streamed_all_hybrid_and_bf16_strict_all_logits() -> Result<()> {
    streamed_model_cases(
        crate::deepseek_v41::DeepseekV41ExpertResidency::StreamedAllExperts,
        crate::deepseek_v41::DeepseekV41EngramResidency::Resident,
        crate::deepseek_v41::DeepseekV41ProjectionResidency::Resident,
        "streamed-all-",
    )
}

#[test]
fn deepseek_v41_streamed_all_tables_hybrid_and_bf16_strict_all_logits() -> Result<()> {
    streamed_model_cases(
        crate::deepseek_v41::DeepseekV41ExpertResidency::StreamedAllExperts,
        crate::deepseek_v41::DeepseekV41EngramResidency::StreamedTables,
        crate::deepseek_v41::DeepseekV41ProjectionResidency::Resident,
        "streamed-tables-",
    )
}

#[test]
fn deepseek_v41_streamed_block_hybrid_and_bf16_strict_all_logits() -> Result<()> {
    streamed_model_cases(
        crate::deepseek_v41::DeepseekV41ExpertResidency::StreamedAllExperts,
        crate::deepseek_v41::DeepseekV41EngramResidency::StreamedTables,
        crate::deepseek_v41::DeepseekV41ProjectionResidency::Streamed,
        "streamed-block-",
    )
}

fn streamed_model_cases(
    residency: crate::deepseek_v41::DeepseekV41ExpertResidency,
    engram_residency: crate::deepseek_v41::DeepseekV41EngramResidency,
    projections: crate::deepseek_v41::DeepseekV41ProjectionResidency,
    prefix: &str,
) -> Result<()> {
    streamed_model_cases_with_vocabulary(
        residency,
        engram_residency,
        projections,
        prefix,
        DeepseekV41VocabularyResidency::Resident,
    )
}

#[test]
fn deepseek_v41_streamed_vocabulary_hybrid_and_bf16_strict_all_logits() -> Result<()> {
    streamed_model_cases_with_vocabulary(
        crate::deepseek_v41::DeepseekV41ExpertResidency::StreamedAllExperts,
        crate::deepseek_v41::DeepseekV41EngramResidency::StreamedTables,
        crate::deepseek_v41::DeepseekV41ProjectionResidency::Streamed,
        "streamed-vocabulary-",
        DeepseekV41VocabularyResidency::Streamed,
    )
}

fn streamed_model_cases_with_vocabulary(
    residency: crate::deepseek_v41::DeepseekV41ExpertResidency,
    engram_residency: crate::deepseek_v41::DeepseekV41EngramResidency,
    projections: crate::deepseek_v41::DeepseekV41ProjectionResidency,
    prefix: &str,
    vocabulary: DeepseekV41VocabularyResidency,
) -> Result<()> {
    packed_model_cases_with_vocabulary(
        include_str!("../../validation/fixtures/deepseek_v41_mixed_model.json"),
        [4141041, 4141042],
        72,
        &format!("{prefix}mixed-model-native-logits.json"),
        false,
        residency,
        engram_residency,
        projections,
        vocabulary,
    )?;
    packed_model_cases_with_vocabulary(
        include_str!("../../validation/fixtures/deepseek_v41_bf16_model.json"),
        [4141061, 4141062],
        103,
        &format!("{prefix}bf16-model-native-logits.json"),
        true,
        residency,
        engram_residency,
        projections,
        vocabulary,
    )
}

#[test]
fn deepseek_v41_bf16_checkpoint_rejects_incomplete_precision_contracts() -> Result<()> {
    let device = VulkanDevice::new()?;
    let f = fixture()?;
    let package = Package::new(&f, false)?;
    let error = DeepseekV41CausalLM::from_checkpoint_with_precision(
        device.clone(),
        &mut package.reader()?,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("BF16 Engram"));

    let source: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_bf16_model.json"
    ))?;
    for (name, expected) in [
        (
            "layers.1.attn.indexer.weights_proj.weight",
            "BF16 indexer requires",
        ),
        (
            "layers.3.attn.compressor.wkv.weight",
            "ratio-1 BF16 compressor requires",
        ),
        ("embed.weight", "must use BF16 storage"),
    ] {
        let mut f: Fixture = serde_json::from_value(source["cases"][0].clone())?;
        let w = f.tensors.get_mut(name).unwrap();
        w.dtype.clear();
        w.values[0] = 0.1;
        let package = Package::new(&f, false)?;
        let error = load(&device, &f, &package).err().unwrap();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
    }
    let f: Fixture = serde_json::from_value(source["cases"][0].clone())?;
    let package = Package::new(&f, false)?;
    let mut reader = package.reader()?;
    let engram = DeepseekV41EngramRuntime::from_checkpoint(
        device.clone(),
        &mut reader,
        f.tokens.clone(),
        f.batch_size,
        f.max_positions,
    )?;
    let error = DeepseekV41Backbone::from_checkpoint_with_precision(
        device,
        &mut reader,
        runtime(&f),
        Some(engram),
        DeepseekV41WeightOutput::Bf16,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("stream precision mismatch"));
    Ok(())
}

fn packed_model_cases(
    source: &str,
    seeds: [u64; 2],
    packed_count: usize,
    output: &str,
    trace: bool,
) -> Result<()> {
    packed_model_cases_with_residency(
        source,
        seeds,
        packed_count,
        output,
        trace,
        crate::deepseek_v41::DeepseekV41ExpertResidency::Resident,
        crate::deepseek_v41::DeepseekV41EngramResidency::Resident,
        crate::deepseek_v41::DeepseekV41ProjectionResidency::Resident,
    )
}

#[allow(clippy::too_many_arguments)]
fn packed_model_cases_with_residency(
    source: &str,
    seeds: [u64; 2],
    packed_count: usize,
    output: &str,
    trace: bool,
    residency: crate::deepseek_v41::DeepseekV41ExpertResidency,
    engram_residency: crate::deepseek_v41::DeepseekV41EngramResidency,
    projections: crate::deepseek_v41::DeepseekV41ProjectionResidency,
) -> Result<()> {
    packed_model_cases_with_vocabulary(
        source,
        seeds,
        packed_count,
        output,
        trace,
        residency,
        engram_residency,
        projections,
        DeepseekV41VocabularyResidency::Resident,
    )
}

fn packed_model_cases_with_vocabulary(
    source: &str,
    seeds: [u64; 2],
    packed_count: usize,
    output: &str,
    trace: bool,
    residency: crate::deepseek_v41::DeepseekV41ExpertResidency,
    engram_residency: crate::deepseek_v41::DeepseekV41EngramResidency,
    projections: crate::deepseek_v41::DeepseekV41ProjectionResidency,
    vocabulary: DeepseekV41VocabularyResidency,
) -> Result<()> {
    #[derive(Deserialize)]
    struct Hybrid {
        cases: Vec<Fixture>,
    }
    let f: Hybrid = serde_json::from_str(source)?;
    assert_eq!(f.cases.len(), 2);
    let binding = format!("{:x}", Sha256::digest(source.as_bytes()));
    let device = VulkanDevice::new()?;
    let mut exported = vec![];
    let mut failures = vec![];
    for (index, (f, seed)) in f.cases.into_iter().zip(seeds).enumerate() {
        assert_eq!(f.seed, seed);
        assert_eq!(f.steps.len(), 6);
        assert_eq!(f.revision, crate::deepseek_v41::REFERENCE_REVISION);
        assert_eq!(
            f.source_sha256,
            "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
        );
        assert_eq!(f.packed_tensors.len(), packed_count);
        if trace {
            assert_eq!(
                f.tensors.values().filter(|t| t.dtype == "bf16").count(),
                if f.stream_precision == "bf16" { 65 } else { 11 }
            );
        }
        let package = Package::new(&f, trace && index == 1)?;
        let before = device.memory_stats()?.live_buffer_bytes;
        let mut reader = package.reader()?;
        let mut model = DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
            device.clone(),
            &mut reader,
            runtime(&f),
            Some(f.tokens.clone()),
            if f.stream_precision == "bf16" {
                DeepseekV41WeightOutput::Bf16
            } else {
                DeepseekV41WeightOutput::Fp32
            },
            residency,
            engram_residency,
            projections,
            vocabulary,
        )?;
        drop(reader);
        let vocabulary_count = if vocabulary == DeepseekV41VocabularyResidency::Resident {
            2
        } else {
            0
        };
        assert_eq!(model.resident_vocabulary_tensor_count(), vocabulary_count);
        let model_buffer_bytes = device.memory_stats()?.live_buffer_bytes - before;
        assert_eq!(
            model.resident_projection_layer_count(),
            if projections == crate::deepseek_v41::DeepseekV41ProjectionResidency::Streamed {
                0
            } else {
                f.config["text_config"]["num_hidden_layers"]
                    .as_u64()
                    .unwrap() as usize
            }
        );
        let resident_count =
            if residency != crate::deepseek_v41::DeepseekV41ExpertResidency::Resident {
                0
            } else {
                f.config["text_config"]["num_hidden_layers"]
                    .as_u64()
                    .unwrap() as usize
                    * f.config["text_config"]["n_routed_experts"]
                        .as_u64()
                        .unwrap() as usize
            };
        assert_eq!(model.resident_routed_expert_count(), resident_count);
        let shared_count =
            if residency == crate::deepseek_v41::DeepseekV41ExpertResidency::StreamedAllExperts {
                0
            } else {
                f.config["text_config"]["num_hidden_layers"]
                    .as_u64()
                    .unwrap() as usize
            };
        assert_eq!(model.resident_shared_expert_count(), shared_count);
        assert_eq!(
            model.stream_precision(),
            if f.stream_precision == "bf16" {
                DeepseekV41WeightOutput::Bf16
            } else {
                DeepseekV41WeightOutput::Fp32
            }
        );
        if trace {
            model.backbone.trace = Some(vec![]);
        }
        let mut steps = vec![];
        for step in &f.steps {
            let result = run(&mut model, step)?;
            assert_eq!(model.resident_vocabulary_tensor_count(), vocabulary_count);
            assert_eq!(model.resident_routed_expert_count(), resident_count);
            let shape = vec![result.batch_size, result.seq_len, result.vocab_size];
            assert_eq!(shape, step.shape);
            let values = result.read_all()?;
            assert!(!values.is_empty());
            assert_eq!(values.len(), step.logits.len());
            let mut worst = 0.0f64;
            for (&a, &b) in values.iter().zip(&step.logits) {
                assert!(a.is_finite() && b.is_finite());
                worst = worst.max((a as f64 - b as f64).abs());
            }
            eprintln!(
                "V4.1 packed full logits seed {seed}, {}: {worst:.17e}",
                step.name
            );
            assert!(
                values.iter().any(|&v| v != 0.0),
                "hybrid fixture must exercise nonzero logits"
            );
            if worst >= 2e-7 {
                failures.push(format!("{seed}, {}: {worst:.17e}", step.name));
            }
            let final_trace = model.final_trace.as_ref().unwrap();
            let width = f.batch_size
                * step.seq
                * f.config["text_config"]["hidden_size"].as_u64().unwrap() as usize;
            let mut exported_step = json!({"name":step.name,"shape":shape,"logits":values,"collapsed":final_trace.0.read_f32(width)?,"normalized":final_trace.1.read_f32(width)?});
            if let Some(stages) = &model.backbone.trace {
                let stages = stages
                    .iter()
                    .map(|(layer, stage, buffer, len)| -> Result<_> {
                        Ok(json!({"layer":layer,"stage":stage,"values":buffer.read_f32(*len)?}))
                    })
                    .collect::<Result<Vec<_>>>()?;
                exported_step["stages"] = json!(stages);
                if f.stream_precision == "bf16" {
                    let mut selections = serde_json::Map::new();
                    for layer in 0..f.config["text_config"]["num_hidden_layers"]
                        .as_u64()
                        .unwrap() as usize
                    {
                        if let Some(s) = model.backbone.index_selection(layer)? {
                            if s.width == 0 {
                                continue;
                            }
                            selections.insert(
                                layer.to_string(),
                                json!({"indices":s.read_indices()?,
                                "score_bits":s.scores.read_u32(s.rows * s.width)?}),
                            );
                        }
                    }
                    exported_step["index_diagnostics"] = json!(selections);
                    let mut caches = serde_json::Map::new();
                    for layer in 0..f.config["text_config"]["num_hidden_layers"]
                        .as_u64()
                        .unwrap() as usize
                    {
                        if let Some((kv, keys)) = model.backbone.attention_cache_trace(layer)? {
                            caches.insert(layer.to_string(), json!({"kv":kv,"keys":keys}));
                        }
                    }
                    exported_step["cache_diagnostics"] = json!(caches);
                }
            }
            steps.push(exported_step);
        }
        exported.push(
            json!({"seed":seed,"revision":f.revision,"fixture_sha256":binding,"steps":steps,
                "expert_residency":residency, "resident_shared_experts":model.resident_shared_expert_count(),
                "engram_residency":engram_residency,"resident_engram_tables":model.resident_engram_table_count(),
                "projection_residency":projections,"resident_projection_layers":model.resident_projection_layer_count(),
                "vocabulary_residency":vocabulary,"resident_vocabulary_tensors":model.resident_vocabulary_tensor_count(),
                "resident_routed_experts":resident_count,"initial_live_buffer_bytes":model_buffer_bytes}),
        );
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference")
        .join(output);
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, serde_json::to_vec(&json!({"cases":exported}))?)?;
    assert!(
        failures.is_empty(),
        "strict <2e-7, rtol=0 failures: {failures:?}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_model_input_rejection_preserves_state_and_partial_failure_requires_prefill(
) -> Result<()> {
    let f = fixture()?;
    let device = VulkanDevice::new()?;
    let package = Package::new(&f, false)?;
    let mut model = load(&device, &f, &package)?;
    assert!(model.decode(&[1, 2], 0, None).is_err());
    assert!(model.prefill(&[1, 2], 0, None).is_err());
    assert!(model.prefill(&[1, 2], 9, None).is_err());
    assert!(model.prefill(&[16, 2], 1, None).is_err());
    assert!(model.prefill(&[1, 2], 1, Some(&[2, 1])).is_err());
    run(&mut model, &f.steps[0])?;
    assert!(model.decode(&[1, 2], 4, None).is_err());
    assert!(model.decode(&[1], 5, None).is_err());
    assert!(model.decode(&[1, 16], 5, None).is_err());
    assert!(model.decode(&[1, 2], 5, Some(&[1])).is_err());
    assert_eq!(model.next_position(), Some(5));
    let mut reference = load(&device, &f, &package)?;
    run(&mut reference, &f.steps[0])?;
    assert_eq!(
        bits(&run(&mut model, &f.steps[1])?.read_all()?),
        bits(&run(&mut reference, &f.steps[1])?.read_all()?)
    );
    // A too-short head fails after the backbone caches have advanced. The
    // language-model wrapper must not expose that as a completed decode step.
    let head = match &mut model.vocabulary {
        VocabularySource::Resident { head, .. } => {
            std::mem::replace(head, GpuBuffer::zeros_f32(&device, 1)?)
        }
        _ => unreachable!("resident test model"),
    };
    assert!(run(&mut model, &f.steps[2]).is_err());
    assert!(run(&mut model, &f.steps[2]).is_err());
    assert_eq!(model.next_position(), None);
    if let VocabularySource::Resident { head: slot, .. } = &mut model.vocabulary {
        *slot = head;
    }
    assert_eq!(
        bits(&run(&mut model, &f.steps[0])?.read_all()?),
        bits(&run(&mut reference, &f.steps[0])?.read_all()?)
    );
    assert_eq!(
        bits(&run(&mut model, &f.steps[1])?.read_all()?),
        bits(&run(&mut reference, &f.steps[1])?.read_all()?)
    );
    Ok(())
}

#[test]
fn deepseek_v41_model_checkpoint_requires_engram_artifacts_and_exact_owned_parameters() -> Result<()>
{
    let mut f = fixture()?;
    let device = VulkanDevice::new()?;
    let package = Package::new(&f, false)?;
    assert!(DeepseekV41CausalLM::from_checkpoint(
        device.clone(),
        &mut package.reader()?,
        runtime(&f),
        None
    )
    .is_err());
    let mut tokens = f.tokens.clone();
    tokens.pad_token_id = 1;
    assert!(DeepseekV41CausalLM::from_checkpoint(
        device.clone(),
        &mut package.reader()?,
        runtime(&f),
        Some(tokens)
    )
    .is_err());
    assert!(DeepseekV41CausalLM::from_checkpoint(
        device.clone(),
        &mut package.reader()?,
        DeepseekV41AttentionRuntimeConfig {
            batch_size: 0,
            ..runtime(&f)
        },
        Some(f.tokens.clone())
    )
    .is_err());
    let mut reader = package.reader()?;
    assert!(reader.load_backbone_layer(6).is_err());
    assert!(reader.load_engram(1).is_err());
    let mut text_only = fixture()?;
    text_only.config["text_config"]["engram_layer_ids"] = json!([]);
    text_only.config["text_config"]["engram_num_embeddings"] = json!([]);
    let plain = Package::new(&text_only, false)?;
    assert!(
        load(&device, &text_only, &plain).is_err(),
        "extra Engram artifacts must be rejected"
    );
    let mut model = DeepseekV41CausalLM::from_checkpoint(
        device.clone(),
        &mut plain.reader()?,
        runtime(&text_only),
        None,
    )?;
    assert!(model.prefill(&[1, 2], 1, Some(&[1, 1])).is_err());
    assert_eq!(model.prefill(&[1, 2], 1, None)?.read_all()?.len(), 32);
    assert_eq!(model.decode(&[3, 4], 1, None)?.read_all()?.len(), 32);
    assert_eq!(
        reader.load_backbone_layer(4)?.hyper.attn_fn,
        f.tensors["layers.4.hc_attn_fn"].values
    );
    assert_eq!(
        reader.load_backbone_layer(4)?.moe.expert_w2[2],
        f.tensors["layers.4.ffn.experts.2.w2.weight"].values
    );
    f.tensors.remove("layers.4.ffn.shared_experts.w2.weight");
    let missing = Package::new(&f, false)?;
    assert!(missing.reader()?.load_backbone_layer(4).is_err());
    f.tensors
        .get_mut("layers.0.engram.embed.weight")
        .unwrap()
        .values[0] = 0.1;
    let bad_table = Package::new(&f, false)?;
    assert!(bad_table.reader()?.load_engram(0).is_err());
    Ok(())
}
