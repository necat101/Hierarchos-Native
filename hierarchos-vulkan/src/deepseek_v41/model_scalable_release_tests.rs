//! Small released-schedule checkpoint through the normal package API.
use super::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf};

fn read<T: DeserializeOwned>(v: &Value) -> Result<T> {
    Ok(serde_json::from_value(v.clone())?)
}

fn fixture() -> Result<(Value, String)> {
    let bytes = include_bytes!("../../validation/fixtures/deepseek_v41_joint_model.json");
    let binding = format!("{:x}", Sha256::digest(bytes));
    assert_eq!(
        binding,
        "00cef75fe55c2243a6c2134c65e4c0adf224ce76791022c97f75738d4a1608b8"
    );
    let all: Value = serde_json::from_slice(bytes)?;
    Ok((all["cases"][0].clone(), binding))
}

#[test]
fn deepseek_v41_scalable_release_preserves_production_feature_contract() -> Result<()> {
    let (f, _) = fixture()?;
    let release: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
    ))?;
    let c = &f["config"]["text_config"];
    let p = &release["text_config"];
    for key in [
        "num_hidden_layers",
        "n_routed_experts",
        "num_experts_per_tok",
        "hc_mult",
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "engram_layer_ids",
        "engram_n_heads",
        "engram_max_ngram_size",
        "candidate_source_layer_id",
        "candidate_block_size",
        "scoring_func",
        "routed_scaling_factor",
        "norm_topk_prob",
    ] {
        assert_eq!(c[key], p[key], "released feature {key}");
    }
    assert_eq!(
        c["compress_ratios"],
        json!(&p["compress_ratios"].as_array().unwrap()[..40])
    );
    assert_eq!(c["hidden_size"], 32);
    assert_eq!(c["vocab_size"], 67);
    assert_eq!(f["steps"].as_array().unwrap().len(), 6);
    assert_eq!(
        f["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["logits"].as_array().unwrap().len())
            .sum::<usize>(),
        3216
    );
    Ok(())
}

#[derive(Deserialize, Serialize)]
struct Route {
    indices: Vec<u32>,
    weight_bits: Vec<u32>,
    counts: Vec<u32>,
}
fn canonical(route: &Route) -> Vec<Vec<(u32, u32)>> {
    assert_eq!(route.indices.len(), route.weight_bits.len());
    assert_eq!(route.indices.len() % 6, 0);
    route
        .indices
        .chunks_exact(6)
        .zip(route.weight_bits.chunks_exact(6))
        .map(|(ids, w)| {
            let mut row: Vec<_> = ids.iter().copied().zip(w.iter().copied()).collect();
            row.sort_unstable();
            assert!(row.windows(2).all(|p| p[0].0 != p[1].0));
            row
        })
        .collect()
}

#[test]
#[ignore = "scalable 40-layer/384-expert release gate; run explicitly before a kernel port"]
fn deepseek_v41_default_package_scalable_release_strict_all_logits() -> Result<()> {
    deepseek_v41_scalable_release_preserves_production_feature_contract()?;
    let (f, binding) = fixture()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation/fixtures")
        .join(f["checkpoint"]["directory"].as_str().unwrap());
    let files: BTreeMap<String, String> = read(&f["checkpoint"]["files"])?;
    for (name, expected) in &files {
        assert_eq!(std::path::Path::new(name).components().count(), 1);
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(root.join(name))?)),
            *expected
        );
    }
    let metadata = serde_json::to_vec(&f["tokens"])?;
    let path = root.join("engram_tokens.json");
    if path.exists() {
        assert_eq!(fs::read(&path)?, metadata);
    } else {
        fs::write(&path, &metadata)?;
    }
    let device = VulkanDevice::new()?;
    let device_name = device.name().to_owned();
    eprintln!("Scalable V4.1 release device: {device_name}");
    let batch: usize = read(&f["batch_size"])?;
    let positions: usize = read(&f["max_positions"])?;
    let mut model = DeepseekV41CausalLM::from_hf_package(device, &root, batch, positions)?;
    let mut steps = vec![];
    let mut total = 0;
    let mut worst = 0f64;
    for step in f["steps"].as_array().unwrap() {
        let ids: Vec<u32> = read(&step["ids"])?;
        let mask: Vec<u32> = read(&step["mask"])?;
        let pos: usize = read(&step["pos"])?;
        let seq: usize = read(&step["seq"])?;
        let expected: Vec<f32> = read(&step["logits"])?;
        let result = if pos == 0 {
            model.prefill(&ids, seq, Some(&mask))?
        } else {
            model.decode(&ids, pos, Some(&mask))?
        };
        let shape = vec![result.batch_size, result.seq_len, result.vocab_size];
        assert_eq!(json!(shape), step["shape"]);
        let logits = result.read_all()?;
        assert!(!logits.is_empty());
        assert_eq!(logits.len(), expected.len());
        let mut drift = 0f64;
        for (&a, &b) in logits.iter().zip(&expected) {
            assert!(a.is_finite() && b.is_finite());
            drift = drift.max((a as f64 - b as f64).abs());
        }
        assert!(
            drift < 2e-7,
            "{} strict logit drift {drift:.17e}",
            step["name"]
        );
        total += logits.len();
        worst = worst.max(drift);
        let mut routing = BTreeMap::new();
        for layer in 0..40 {
            let (w, ids, counts) = model
                .backbone
                .moe_route_trace(layer)
                .context("missing released routing")?;
            let route = Route {
                indices: ids.read_u32(batch * seq * 6)?,
                weight_bits: w.read_u32(batch * seq * 6)?,
                counts: counts.read_u32(384)?,
            };
            let expected: Route = read(&step["routing"][layer.to_string()])?;
            assert_eq!(route.counts, expected.counts);
            assert_eq!(canonical(&route), canonical(&expected));
            routing.insert(layer.to_string(), route);
        }
        assert_eq!(model.next_position(), Some(pos + seq));
        assert_eq!(model.resident_routed_expert_count(), 0);
        assert_eq!(model.resident_shared_expert_count(), 0);
        assert_eq!(model.resident_engram_table_count(), 0);
        assert_eq!(model.resident_projection_layer_count(), 0);
        assert_eq!(model.resident_vocabulary_tensor_count(), 0);
        eprintln!(
            "Scalable V4.1 {}: {} logits, drift {drift:.17e}",
            step["name"],
            logits.len()
        );
        steps.push(json!({"name":step["name"],"shape":shape,"logits":logits,"routing":routing}));
    }
    assert_eq!(total, 3216);
    let output = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/scalable-release-native-logits.json");
    fs::write(
        output,
        serde_json::to_vec(&json!({"cases":[{
            "seed":f["seed"],"revision":f["revision"],"fixture_sha256":binding,
            "checkpoint_files":files,"engram_metadata_sha256":format!("{:x}",Sha256::digest(metadata)),
            "loader":"from_hf_package","device":device_name,"full_logits":total,"max_abs_drift":worst,"steps":steps
        }]}))?,
    )?;
    Ok(())
}
