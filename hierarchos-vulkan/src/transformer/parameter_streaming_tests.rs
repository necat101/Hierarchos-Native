mod parameter_streaming {
    use super::*;
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let Ok(path)=self.0.canonicalize() else{return};
            let Ok(parent)=std::env::temp_dir().canonicalize() else{return};
            if path.parent()==Some(parent.as_path()) && path.file_name().is_some_and(|n|n.to_string_lossy().starts_with("hierarchos-linear-stream-")) {let _=fs::remove_dir_all(path);}
        }
    }

    /// One architecture in the streaming matrix. `train` marks the families
    /// whose native backward and AdamW path is already qualified; the others
    /// are still checked for resident/streamed inference identity, which is
    /// the part of the contract that does not depend on a trainer.
    struct Case {
        family: &'static str,
        config: serde_json::Value,
        graph: VulkanTransformer,
        train: bool,
    }

    fn run_case(device: &VulkanDevice, case: Case, source_first: bool) -> Result<()> {
        let Case { family, config, graph: source, train } = case;
        let root=Scratch(std::env::temp_dir().join(format!("hierarchos-linear-stream-{}-{}-{}",family,std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos())));
        let metadata=root.0.join("source");let package=root.0.join("package");fs::create_dir_all(&metadata)?;
        fs::write(metadata.join("config.json"),serde_json::to_vec(&config)?)?;
        source.export_hf_package(&metadata,&package)?;drop(source);
        let mut resident=VulkanTransformer::from_hf_package(device.clone(),&package,1,8)?;
        let mut streamed=if source_first {
            let config=VulkanTransformerConfig::from_hf_config(package.join("config.json"))?;
            assert!(streamed_load::supports(&package,&config)?);
            VulkanTransformer::from_hf_package_with_streamed_parameters(device.clone(),&package,config,1,8)?
        } else {VulkanTransformer::from_hf_package(device.clone(),&package,1,8)?};
        let ids=[3u32,11,7,13,5,19,2,17];let mask=[1.0f32;8];let targets=[11u32,7,13,5,19,2,17,3];
        let before=device.memory_stats()?.live_buffer_bytes;
        let count=if source_first {streamed.streamed_linears} else {
            streamed.stream_linear_parameters_from_package(&package)
                .with_context(||format!("{family}: package-backed graph linears"))?
        };
        let after=device.memory_stats()?.live_buffer_bytes;
        assert!(count>0 && (source_first || after<before),"{family}: no base residency retired");
        if source_first {
            assert!(streamed.parameters_streamed_during_loading());
            assert_eq!(streamed.streamed_linear_package_io()?.len(),1);
            streamed.for_each_frozen_graph_linear_mut(|_,_,linear| {
                assert!(linear.weight.tiered_source().is_some());
                assert_eq!(linear.weight.training_state_bytes(),0);
                Ok(())
            })?;
        }
        assert_eq!(streamed.forward_logits(&ids,&mask)?,resident.forward_logits(&ids,&mask)?,"{family} base logits");
        if !train {
            eprintln!("graph parameter streaming {family}: {count} linears, retired {} bytes; base logits bit-exact (inference)",before-after);
            return Ok(());
        }
        let cfg=VulkanTransformerLoraConfig::causal_lm(2,4,vec!["all-linear".into()],family=="gpt2")?;
        resident.enable_lora(cfg.clone(),187)?;streamed.enable_lora(cfg,187)?;
        for step in 0..2 {
            assert_eq!(streamed.forward_logits(&ids,&mask)?,resident.forward_logits(&ids,&mask)?,"{family} PEFT logits {step}");
            let hyper=AdamWHyperParams{lr:1e-3,beta1:0.9,beta2:0.999,eps:1e-8,weight_decay:0.01};
            let a=resident.train_step(&ids,&targets,&mask,&mask,hyper)?;
            let b=streamed.train_step(&ids,&targets,&mask,&mask,hyper)?;
            assert_eq!(a.loss.to_bits(),b.loss.to_bits(),"{family} loss {step}");
        }
        let a=root.0.join("resident-adapter");let b=root.0.join("streamed-adapter");resident.export_lora_adapter(&a)?;streamed.export_lora_adapter(&b)?;
        let left=fs::read(a.join("adapter_model.safetensors"))?;let right=fs::read(b.join("adapter_model.safetensors"))?;
        let left=SafeTensors::deserialize(&left)?;let right=SafeTensors::deserialize(&right)?;
        for name in left.names(){assert_eq!(left.tensor(name)?.data(),right.tensor(name)?.data(),"{family} {name}");}
        eprintln!("graph parameter streaming {family}: {count} linears, retired {} bytes; base/PEFT logits and two-step AdamW factors bit-exact",before-after);
        Ok(())
    }

    /// Streaming is a property of the shared graph traversal, so the matrix
    /// spans the loader shapes that reach it: RMSNorm dense, Conv1D, tied and
    /// untied vocabulary, packed QKV with bias, and routed-expert MoE.
    #[test]
    fn graph_parameter_streaming_preserves_inference_lora_backward_and_adamw() -> Result<()> {
        let device=VulkanDevice::new()?;
        let llama=tiny_mistral_config();
        let tied=tiny_tied_llama_config();
        let cases=vec![
            Case{family:"llama",config:llama.clone(),graph:tiny_llama_layout_vulkan_graph(device.clone(),llama)?,train:true},
            Case{family:"mistral",config:tiny_mistral_config(),graph:tiny_mistral_vulkan_graph(device.clone())?,train:true},
            Case{family:"gpt2",config:tiny_gpt2_config(),graph:tiny_gpt2_vulkan_graph(device.clone())?,train:true},
            Case{family:"mixtral",config:tiny_mixtral_config(),graph:tiny_mixtral_vulkan_graph(device.clone())?,train:true},
            Case{family:"qwen2",config:tiny_qwen2_config(),graph:tiny_qwen2_vulkan_graph(device.clone())?,train:true},
            Case{family:"tied-llama",config:tied.clone(),graph:tiny_llama_layout_vulkan_graph(device.clone(),tied)?,train:true},
            Case{family:"jais2",config:tiny_jais2_config(),graph:tiny_jais2_vulkan_graph(device.clone())?,train:true},
            Case{family:"gpt-oss",config:tiny_gpt_oss_config(),graph:tiny_gpt_oss_vulkan_graph(device.clone())?,train:false},
            Case{family:"glm5-next-linear-moe",config:tiny_glm5_next_linear_moe_config(),graph:tiny_glm5_next_linear_moe_vulkan_graph(device.clone())?,train:false},
        ];
        for case in cases { run_case(&device,case,false)?; }
        Ok(())
    }

    #[test]
    fn source_first_dense_and_moe_preserve_logits_lora_and_adamw() -> Result<()> {
        let device=VulkanDevice::new()?;
        let mut llama=tiny_mistral_config();
        llama["model_type"]=serde_json::json!("llama");
        let tied=tiny_tied_llama_config();
        let cases=vec![
            Case{family:"llama",config:llama.clone(),graph:tiny_llama_layout_vulkan_graph(device.clone(),llama)?,train:true},
            Case{family:"mistral",config:tiny_mistral_config(),graph:tiny_mistral_vulkan_graph(device.clone())?,train:true},
            Case{family:"mixtral",config:tiny_mixtral_config(),graph:tiny_mixtral_vulkan_graph(device.clone())?,train:true},
            Case{family:"qwen2",config:tiny_qwen2_config(),graph:tiny_qwen2_vulkan_graph(device.clone())?,train:true},
            Case{family:"tied-llama",config:tied.clone(),graph:tiny_llama_layout_vulkan_graph(device.clone(),tied)?,train:true},
        ];
        for case in cases {run_case(&device,case,true)?;}
        Ok(())
    }

    #[test]
    fn single_tile_projection_and_input_adjoint_preserve_aliased_buffers() -> Result<()> {
        let device=VulkanDevice::new()?;
        let kernels=TransformerKernels::new(&device)?;
        let root=Scratch(std::env::temp_dir().join(format!("hierarchos-linear-stream-alias-{}",std::process::id())));
        fs::create_dir_all(&root.0)?;
        let d=16usize;let rows=8usize;
        let weights=(0..d*d).map(|n|((n*37%97) as f32-48.0)*0.0009).collect::<Vec<_>>();
        let bias=(0..d).map(|n|n as f32*0.001-0.008).collect::<Vec<_>>();
        let view=TensorView::new(Dtype::F32,vec![d,d],bytemuck::cast_slice(&weights))?;
        serialize_to_file([("w",view)],None,&root.0.join("model.safetensors"))?;
        let mut store=Gpt2TensorStore::load(&root.0)?;
        let source=store.take_expected_source("w",&[d,d])?;
        let streamed=TieredLmHead::from_source(&device,&source,rows,Some(bias.clone()))?;
        assert!(streamed.single_tile());
        let resident=VulkanLinear::new(&device,d,d,&weights,&bias)?;
        let input_values=(0..rows*d).map(|n|((n*17%71) as f32-35.0)*0.017).collect::<Vec<_>>();
        let input=GpuBuffer::from_f32(&device,&input_values)?;
        let expected=GpuBuffer::zeros_f32(&device,rows*d)?;
        let mut commands=vulkan::ComputeBatch::new(&device)?;
        resident.record_forward(&mut commands,&kernels,&input,&expected,rows)?;
        commands.submit()?;
        let expected=expected.read_f32(rows*d)?;
        let output=GpuBuffer::zeros_f32(&device,rows*d)?;
        let mut commands=vulkan::ComputeBatch::new(&device)?;
        streamed.record_projection(&mut commands,&kernels,&input,&output,rows,false,false)?;
        commands.submit()?;
        assert_eq!(output.read_f32(rows*d)?,expected);
        let alias=GpuBuffer::from_f32(&device,&input_values)?;
        let mut commands=vulkan::ComputeBatch::new(&device)?;
        streamed.record_projection(&mut commands,&kernels,&alias,&alias,rows,false,false)?;
        commands.submit()?;
        assert_eq!(alias.read_f32(rows*d)?,expected,"in-place forward must use scratch");
        let expected_grad=GpuBuffer::zeros_f32(&device,rows*d)?;
        let mut commands=vulkan::ComputeBatch::new(&device)?;
        streamed.record_input_gradient(&mut commands,&input,&expected_grad,rows)?;
        commands.submit()?;
        let alias=GpuBuffer::from_f32(&device,&input_values)?;
        let mut commands=vulkan::ComputeBatch::new(&device)?;
        streamed.record_input_gradient(&mut commands,&alias,&alias,rows)?;
        commands.submit()?;
        assert_eq!(alias.read_f32(rows*d)?,expected_grad.read_f32(rows*d)?,"in-place adjoint must pack its input");
        Ok(())
    }

    /// The tied variant differs only in `tie_word_embeddings`, so both shapes
    /// load the same layers while the vocabulary identity is shared or split.
    fn tiny_tied_llama_config() -> serde_json::Value {
        let mut value=tiny_mistral_config();
        value["model_type"]=serde_json::json!("llama");
        value["tie_word_embeddings"]=serde_json::json!(true);
        if let Some(object)=value.as_object_mut(){object.remove("sliding_window");}
        value
    }
}
