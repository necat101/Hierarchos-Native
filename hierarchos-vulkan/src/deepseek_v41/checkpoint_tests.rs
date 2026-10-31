use super::*;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Package {
    root: PathBuf,
}

impl Package {
    fn new() -> Result<Self> {
        static ID: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hierarchos-v41-checkpoint-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        let mut config: Value = serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
        ))?;
        config["text_config"]["hidden_size"] = serde_json::json!(8);
        config["text_config"]["head_dim"] = serde_json::json!(16);
        config["text_config"]["qk_rope_head_dim"] = serde_json::json!(8);
        fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
        Ok(Self { root })
    }

    fn write(
        &self,
        filename: &str,
        name: &str,
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<()> {
        serialize_to_file(
            vec![(name, TensorView::new(dtype, shape, bytes)?)],
            None,
            &self.root.join(filename),
        )?;
        Ok(())
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        // Only remove the individual files created inside this unique test
        // directory; no recursive deletion or traversal of referenced shards.
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let _ = fs::remove_dir(&self.root);
    }
}

#[test]
fn deepseek_v41_checkpoint_reads_sharded_f32_bf16_and_actual_compressor_keys() -> Result<()> {
    let package = Package::new()?;
    let kv = vec![0.25f32; 128];
    let gate = vec![-0.5f32; 128];
    let norm = vec![1.0f32; 16];
    let kv_bytes = bytemuck::cast_slice(&kv);
    let gate_bytes = bytemuck::cast_slice(&gate);
    let norm_bytes = bytemuck::cast_slice(&norm);
    let entries = vec![
        (
            "layers.2.attn.compressor.wkv.weight",
            TensorView::new(Dtype::F32, vec![16, 8], kv_bytes)?,
        ),
        (
            "layers.2.attn.compressor.wgate.weight",
            TensorView::new(Dtype::F32, vec![16, 8], gate_bytes)?,
        ),
        (
            "layers.2.attn.compressor.norm.weight",
            TensorView::new(Dtype::F32, vec![16], norm_bytes)?,
        ),
    ];
    serialize_to_file(entries, None, &package.root.join("first.safetensors"))?;
    let bf_kv: Vec<u8> = [0x80u8, 0x3e].repeat(128);
    let bf_norm: Vec<u8> = [0x80u8, 0x3f].repeat(16);
    serialize_to_file(
        vec![
            (
                "layers.20.attn.compressor.wkv.weight",
                TensorView::new(Dtype::BF16, vec![16, 8], &bf_kv)?,
            ),
            (
                "layers.20.attn.compressor.norm.weight",
                TensorView::new(Dtype::BF16, vec![16], &bf_norm)?,
            ),
        ],
        None,
        &package.root.join("second.safetensors"),
    )?;
    let index = serde_json::json!({"weight_map": {
        "layers.2.attn.compressor.wkv.weight": "first.safetensors", "layers.2.attn.compressor.wgate.weight": "first.safetensors",
        "layers.2.attn.compressor.norm.weight": "first.safetensors", "layers.20.attn.compressor.wkv.weight": "second.safetensors",
        "layers.20.attn.compressor.norm.weight": "second.safetensors"
    }});
    fs::write(
        package.root.join("model.safetensors.index.json"),
        serde_json::to_vec(&index)?,
    )?;
    let mut reader = DeepseekV41Checkpoint::open(&package.root)?;
    let c2 = reader.load_compressor(2)?;
    assert_eq!(c2.kv_weight, kv);
    assert_eq!(c2.gate_weight, Some(gate));
    assert_eq!(c2.norm_weight, norm);
    assert_eq!(c2.compress_ratio, 2);
    let c1 = reader.load_compressor(20)?;
    assert_eq!(c1.kv_weight, kv);
    assert!(c1.gate_weight.is_none());
    assert_eq!(c1.compress_ratio, 1);
    assert!(reader.load_compressor(24).is_err());
    assert!(reader.load_compressor(40).is_err());
    assert!(reader
        .read_tensor("layers.2.attn.compressor.wkv.weight", &[8, 16])
        .is_err());
    assert!(reader.read_tensor("missing", &[1]).is_err());
    Ok(())
}

#[test]
fn deepseek_v41_checkpoint_rejects_packed_nonfinite_and_invalid_offsets() -> Result<()> {
    let package = Package::new()?;
    package.write("model.safetensors", "packed", Dtype::I8, vec![2], &[0, 1])?;
    let mut reader = DeepseekV41Checkpoint::open(&package.root)?;
    let error = reader.read_tensor("packed", &[2]).unwrap_err().to_string();
    assert!(error.contains("requires the explicit packed-weight API"));
    package.write(
        "model.safetensors",
        "bad",
        Dtype::F32,
        vec![1],
        &f32::NAN.to_le_bytes(),
    )?;
    let mut reader = DeepseekV41Checkpoint::open(&package.root)?;
    assert!(reader
        .read_tensor("bad", &[1])
        .unwrap_err()
        .to_string()
        .contains("non-finite"));
    let header = br#"{"bad":{"dtype":"F32","shape":[1],"data_offsets":[0,4000]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0u8; 4]);
    fs::write(package.root.join("model.safetensors"), bytes)?;
    let mut reader = DeepseekV41Checkpoint::open(&package.root)?;
    assert!(reader
        .read_tensor("bad", &[1])
        .unwrap_err()
        .to_string()
        .contains("offsets"));
    fs::write(
        package.root.join("model.safetensors"),
        u64::MAX.to_le_bytes(),
    )?;
    let mut reader = DeepseekV41Checkpoint::open(&package.root)?;
    assert!(reader.read_tensor("bad", &[1]).is_err());
    Ok(())
}

#[test]
fn deepseek_v41_checkpoint_rejects_shard_path_escape() -> Result<()> {
    let package = Package::new()?;
    for filename in ["../model.safetensors", "/model.safetensors"] {
        let index = serde_json::json!({"weight_map": {"bad": filename}});
        fs::write(
            package.root.join("model.safetensors.index.json"),
            serde_json::to_vec(&index)?,
        )?;
        let mut reader = DeepseekV41Checkpoint::open(&package.root)?;
        assert!(reader
            .read_tensor("bad", &[1])
            .unwrap_err()
            .to_string()
            .contains("relative path"));
    }
    Ok(())
}

#[test]
fn deepseek_v41_checkpoint_clones_share_index_and_shard_headers() -> Result<()> {
    let package = Package::new()?;
    package.write(
        "data.safetensors",
        "weight",
        Dtype::F32,
        vec![2],
        bytemuck::cast_slice(&[0.25f32, -0.5]),
    )?;
    fs::write(
        package.root.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map":{"weight":"data.safetensors"}}))?,
    )?;
    let mut reader = DeepseekV41Checkpoint::open(&package.root)?;
    assert_eq!(reader.read_tensor("weight", &[2])?, [0.25, -0.5]);
    let mut cloned = reader.clone();
    assert!(reader.shares_metadata_with(&cloned));
    drop(reader);
    assert_eq!(cloned.read_tensor("weight", &[2])?, [0.25, -0.5]);
    Ok(())
}

#[test]
fn lora_targets_cover_only_additive_projection_boundaries() {
    use super::checkpoint::lora_target_supported;
    for target in [
        "layers.3.attn.wq_a",
        "layers.3.attn.wq_b",
        "layers.3.attn.wkv",
        "layers.3.attn.wo_b",
        "layers.3.engram.wkv",
        "layers.3.indexer.wq_b",
        "layers.3.indexer.weights_proj",
        "layers.3.indexer.wk",
        "layers.3.ffn.shared_experts.w1",
        "layers.3.ffn.experts.7.w2",
    ] {
        assert!(lora_target_supported(target), "{target}");
    }
    for target in [
        "layers.3.attn.wo_a",
        "layers.3.attn.q_norm",
        "layers.3.attn.compressor.wkv",
        "layers.3.ffn.gate",
        "layers.3.attn.wq",
        "layers.3.attn.wq_a.extra",
        "",
    ] {
        assert!(!lora_target_supported(target), "{target}");
    }
}
