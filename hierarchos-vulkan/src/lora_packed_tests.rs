#[test]
fn packed_parameter_layouts_match_independent_einsum_and_reverse_with_offsets() -> Result<()> {
    let oracle: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../validation/fixtures/packed_lora_layouts.json"
    ))?;
    let cases = oracle["cases"].as_array().unwrap();
    let root = temp_dir("packed-layout-oracles");
    fs::create_dir_all(root.join("adapter"))?;
    for pair in cases.chunks(2) {
        let mut base = vec![("000.guard", vec![11], vec![9.5; 11])];
        let mut adapter = Vec::new();
        let mut layouts = serde_json::Map::new();
        for (index, case) in pair.iter().enumerate() {
            let key = if index == 0 {
                "layer.experts.first"
            } else {
                "layer.experts.second"
            };
            base.push((
                key,
                serde_json::from_value(case["shape"].clone())?,
                serde_json::from_value(case["base"].clone())?,
            ));
            let prefix = if index == 0 {
                "base_model.model.layer.experts.base_layer"
            } else {
                "base_model.model.layer.experts"
            };
            adapter.push((
                format!("{prefix}.lora_A.weight"),
                vec![6, 2],
                serde_json::from_value::<Vec<f32>>(case["a"].clone())?,
            ));
            adapter.push((
                format!("{prefix}.lora_B.weight"),
                vec![case["output"].as_u64().unwrap() as usize, 6],
                serde_json::from_value(case["b"].clone())?,
            ));
            if case["input"] == case["output"] {
                layouts.insert(key.into(), case["layout"].clone());
            }
        }
        let base_file = root.join("base.safetensors");
        write_tensors(&base_file, &base)?;
        let tensors = adapter
            .iter()
            .map(|(name, shape, values)| (name.as_str(), shape.clone(), values.clone()))
            .collect::<Vec<_>>();
        write_tensors(&root.join("adapter/adapter_model.safetensors"), &tensors)?;
        let mut config = json!({"peft_type":"LORA","task_type":"FEATURE_EXTRACTION","r":2,"lora_alpha":3,
            "target_modules":null,"target_parameters":["first","second"],"bias":"none"});
        if !layouts.is_empty() {
            fs::write(
                root.join("adapter/adapter_config.json"),
                serde_json::to_vec(&config)?,
            )?;
            let error = merge_peft_lora_safetensors(
                &base_file,
                &root.join("adapter"),
                &root.join("ambiguous.safetensors"),
            )
            .unwrap_err();
            assert!(format!("{error:#}").contains("ambiguous orientation"));
            config["target_parameter_layouts"] = layouts.into();
        }
        fs::write(
            root.join("adapter/adapter_config.json"),
            serde_json::to_vec(&config)?,
        )?;
        let merged = root.join("merged.safetensors");
        let restored = root.join("restored.safetensors");
        merge_peft_lora_safetensors(&base_file, &root.join("adapter"), &merged)?;
        unmerge_peft_lora_safetensors(&merged, &root.join("adapter"), &restored)?;
        for (index, case) in pair.iter().enumerate() {
            let key = if index == 0 {
                "layer.experts.first"
            } else {
                "layer.experts.second"
            };
            let (_, actual) = crate::checkpoint::read_f32_tensor(&merged, key)?;
            assert_eq!(
                actual,
                serde_json::from_value::<Vec<f32>>(case["merged"].clone())?,
                "{}",
                case["name"]
            );
            let (_, actual) = crate::checkpoint::read_f32_tensor(&restored, key)?;
            // The frozen inputs/deltas are dyadic and exactly representable;
            // this is not a claim of universal bitwise floating-point reversal.
            assert_eq!(
                actual,
                serde_json::from_value::<Vec<f32>>(case["base"].clone())?
            );
        }
        assert_eq!(
            crate::checkpoint::read_f32_tensor(&merged, "000.guard")?.1,
            vec![9.5; 11]
        );
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
