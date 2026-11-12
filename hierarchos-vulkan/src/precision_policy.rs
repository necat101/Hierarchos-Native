//! Shared CLI/desktop admission policy. This file has no Vulkan dependency so
//! the desktop can inspect a local package without constructing a model.
use serde_json::Value;
use std::path::Path;

pub const FULL_FP16_MAX_REDUCTION: usize = 64;

pub fn full_fp16_hardware_qualified(vendor: u32, device: u32, key: &str) -> bool {
    vendor == 0x1002
        && device == 0x15bf
        && key == "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]"
}

pub fn check_full_fp16_text_profile(
    config: &Value,
    batch: usize,
    seq: usize,
) -> Result<(), String> {
    let rows = batch
        .checked_mul(seq)
        .ok_or("Full FP16 row count overflow")?;
    if config["model_type"] != "llama"
        || config["hidden_size"] != 64
        || config["intermediate_size"] != 64
        || config["vocab_size"] != 64
        || config["num_hidden_layers"] != 1
        || config["num_attention_heads"] != 8
        || config["num_key_value_heads"] != 8
        || config["head_dim"].as_u64().unwrap_or(8) != 8
        || config["attention_dropout"].as_f64().unwrap_or(0.0) != 0.0
        || config["tie_word_embeddings"].as_bool().unwrap_or(false)
        || !(33..=64).contains(&rows)
        || seq > 64
    {
        return Err("Full FP16 is qualified for the 64-wide, one-layer Llama profile (64 intermediate/vocabulary, eight heads, no dropout, 33–64 batch×sequence rows). Other model profiles require independent qualification.".into());
    }
    Ok(())
}

pub fn check_full_fp16_package(
    root: &Path,
    batch: usize,
    seq: usize,
    vision: bool,
) -> Result<(), String> {
    let read = |path: &Path| -> Result<Value, String> {
        serde_json::from_slice(
            &std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?,
        )
        .map_err(|e| format!("decode {}: {e}", path.display()))
    };
    let config = read(&root.join("config.json"))?;
    if !vision {
        return check_full_fp16_text_profile(&config, batch, seq);
    }
    if config["model_type"] != "hierarchos_image_text" || batch != 2 || seq != 24 {
        return Err("Full FP16 image training is qualified for a shared CLIP64/Llama64 package with batch 2 and sequence length 24. Other image profiles remain unavailable.".into());
    }
    check_full_fp16_text_profile(&read(&root.join("text/config.json"))?, batch, seq)?;
    let image = read(&root.join("image/config.json"))?;
    let v = &image["vision_config"];
    if v["hidden_size"] != 64
        || v["intermediate_size"] != 64
        || v["num_hidden_layers"] != 1
        || v["num_attention_heads"] != 8
        || v["image_size"] != 16
        || v["patch_size"] != 4
        || v["attention_dropout"].as_f64().unwrap_or(0.0) != 0.0
        || image["image_token_index"] != 31
        || image["vision_feature_layer"] != -1
        || image["projector_hidden_act"] != "gelu"
    {
        return Err("Full FP16 requires the qualified CLIP64 image profile (one layer, eight heads, 16×16 image, 4×4 patch, GELU projector).".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_profile_requires_the_qualified_geometry_and_driver() {
        let config = serde_json::json!({"model_type":"llama","hidden_size":64,"intermediate_size":64,
            "vocab_size":64,"num_hidden_layers":1,"num_attention_heads":8,"num_key_value_heads":8});
        assert!(check_full_fp16_text_profile(&config, 2, 24).is_ok());
        assert!(check_full_fp16_text_profile(&config, 1, 24).is_err());
        let mut wide = config.clone();
        wide["hidden_size"] = 128.into();
        assert!(check_full_fp16_text_profile(&wide, 2, 24).is_err());
        assert!(full_fp16_hardware_qualified(
            0x1002,
            0x15bf,
            "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]"
        ));
        assert!(!full_fp16_hardware_qualified(
            0x1002,
            0x15bf,
            "another driver"
        ));
    }
}
