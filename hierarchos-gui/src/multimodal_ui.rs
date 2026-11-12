use super::*;

impl NativeApp {
    pub(super) fn backend_check_args(&self)->Vec<String> {
        if self.mode!=Workflow::ImageGenerate{return vec!["doctor".into()];}
        let mut args=vec!["image-engine-info".into()];if !self.image_engine.trim().is_empty(){args.extend(["--engine".into(),self.image_engine.trim().into()]);}args
    }
    pub(super) fn build_multimodal_args(&self)->Result<Vec<String>,String> {
        let mut args=vec![self.mode.cli_mode().to_owned()];
        let model=self.model.trim();
        if self.mode==Workflow::ImageGenerate {require_file(model,"image model checkpoint")?;}else{require_directory(model,"vision model package")?;}
        args.extend(["--model-path".into(),model.into()]);
        if !self.device_index.trim().is_empty(){let _:usize=self.device_index.trim().parse().map_err(|_|"Device index must be a non-negative integer.")?;args.extend(["--device-index".into(),self.device_index.trim().into()]);}
        if self.mode==Workflow::ImageGenerate {
            if self.prompt.trim().is_empty(){return Err("Enter a text prompt for the image.".into());}
            args.extend(["--prompt".into(),self.prompt.trim().into()]);
            let output=self.output.trim();if output.is_empty()||!output.to_ascii_lowercase().ends_with(".png"){return Err("Choose a new output PNG filename.".into());}
            if Path::new(output).exists(){return Err("Choose a new output PNG; existing images are preserved.".into());}
            args.extend(["--output".into(),output.into()]);
            for (name,value) in [("--width",&self.image_width),("--height",&self.image_height)] {
                let n:usize=value.trim().parse().map_err(|_|format!("{name} must be an integer"))?;
                if n<64||n>4096||n%64!=0{return Err(format!("{name} must be a multiple of 64 between 64 and 4096."));}
                args.extend([name.into(),n.to_string()]);
            }
            push_positive(&mut args,"--steps",&self.image_steps)?;
            push_positive_float(&mut args,"--cfg-scale",&self.image_cfg)?;
            let seed:i64=self.image_seed.trim().parse().map_err(|_|"Seed must be an integer (-1 chooses a random seed).")?;
            if seed< -1{return Err("Seed must be -1 or a non-negative integer.".into());}
            args.extend(["--seed".into(),seed.to_string()]);
            if !self.image_engine.trim().is_empty(){args.extend(["--engine".into(),self.image_engine.trim().into()]);}
            if !self.image_negative.trim().is_empty(){args.extend(["--negative-prompt".into(),self.image_negative.trim().into()]);}
            if !self.image_vae.trim().is_empty(){require_file(self.image_vae.trim(),"VAE checkpoint")?;args.extend(["--vae".into(),self.image_vae.trim().into()]);}
            return Ok(args);
        }
        push_positive(&mut args,"--max-length",&self.seq_len)?;
        if !self.tokenizer.trim().is_empty(){require_existing(self.tokenizer.trim(),"local tokenizer")?;args.extend(["--tokenizer-path".into(),self.tokenizer.trim().into()]);}
        if self.mode==Workflow::VisionTrain {
            require_file(self.dataset.trim(),"image/text JSONL dataset")?;args.extend(["--train".into(),self.dataset.trim().into()]);
            if self.output.trim().is_empty(){return Err("Choose a new or empty training output directory.".into());}
            let output=Path::new(self.output.trim());if output.exists()&&(!output.is_dir()||std::fs::read_dir(output).map_err(|e|e.to_string())?.next().is_some()){return Err("Choose a new or empty training output directory.".into());}
            args.extend(["--out-dir".into(),self.output.trim().into()]);
            push_positive(&mut args,"--epochs",&self.epochs)?;push_positive(&mut args,"--batch-size",&self.batch_size)?;push_positive_float(&mut args,"--lr",&self.learning_rate)?;
            if !self.vision_resume.trim().is_empty(){require_directory(self.vision_resume.trim(),"native V4.1 resume checkpoint")?;args.extend(["--resume".into(),self.vision_resume.trim().into()]);}
        }else {
            require_file(self.vision_image.trim(),"input image")?;
            if self.prompt.trim().is_empty(){return Err("Enter a prompt about the input image.".into());}
            args.extend(["--image".into(),self.vision_image.trim().into(),"--prompt".into(),self.prompt.trim().into()]);
            push_positive(&mut args,"--max-new-tokens",&self.max_new_tokens)?;
            if self.do_sample {args.push("--do-sample".into());push_positive_float(&mut args,"--temperature",&self.temperature)?;let k:usize=self.top_k.trim().parse().map_err(|_|"Top-k must be non-negative.")?;args.extend(["--top-k".into(),k.to_string(),"--top-p".into(),parse_probability("--top-p",&self.top_p)?.to_string()]);}
            else {args.push("--no-do-sample".into());}
            if !self.vision_state.trim().is_empty(){require_directory(self.vision_state.trim(),"native V4.1 image export")?;args.extend(["--native-state".into(),self.vision_state.trim().into()]);}
        }
        let config:serde_json::Value=std::fs::read(Path::new(model).join("config.json")).ok().and_then(|b|serde_json::from_slice(&b).ok()).ok_or("Choose a vision package with a valid config.json.")?;
        let native=config["model_type"].as_str()==Some("deepseek_v41");
        if self.mode == Workflow::VisionTrain {
            if native && self.wmma_mode == WmmaModeChoice::Mixed {
                return Err("Native DeepSeek V4.1 image training currently uses FP32 kernels. Choose FP32 training precision.".into());
            }
            self.push_wmma_args(&mut args)?;
        }
        if !matches!(config["model_type"].as_str(),Some("hierarchos_image_text"|"llava"|"gemma3"|"gemma4"|"deepseek_v41")){return Err("Select a native vision model or composed hierarchos_image_text package; ordinary text packages need an image tower/projector first.".into());}
        if native {
            if self.mode==Workflow::VisionTrain {if self.batch_size.trim()!="1"{return Err("Native V4.1 image training currently requires batch size 1.".into());}push_positive(&mut args,"--lora-rank",&self.lora_rank)?;}
            if self.stream_parameters {args.push("--stream-parameters".into());}
        }else if !self.vision_state.trim().is_empty()&&self.mode==Workflow::VisionGenerate||!self.vision_resume.trim().is_empty()&&self.mode==Workflow::VisionTrain {
            return Err("Separate image state/resume is for native V4.1. For other families choose the exported model/ package.".into());
        }
        if config["model_type"].as_str()==Some("gemma4")&&self.mode==Workflow::VisionTrain&&self.batch_size.trim()!="1"{return Err("Native Gemma4 image training currently requires batch size 1.".into());}
        Ok(args)
    }

    pub(super) fn multimodal_form(&mut self,ui:&mut egui::Ui) {
        let image_gen=self.mode==Workflow::ImageGenerate;
        path_row(ui,if image_gen{"Image model checkpoint"}else{"Vision model package"},&mut self.model,true,if image_gen{FileChoice::File}else{FileChoice::Folder});
        if image_gen {
            ui.small("Generate a PNG with a Stable Diffusion single-file image checkpoint and a native Vulkan sd-cli engine.");
            path_row(ui,"Image engine (optional)",&mut self.image_engine,true,FileChoice::File);
            ui.small("Blank uses HIERARCHOS_SD_CLI, a colocated sd-cli, or sd-cli on PATH.");
            path_row(ui,"VAE override (optional)",&mut self.image_vae,true,FileChoice::File);
            path_row(ui,"Output PNG",&mut self.output,true,FileChoice::SaveImage);
            ui.label("Prompt");ui.add(egui::TextEdit::multiline(&mut self.prompt).desired_width(f32::INFINITY).desired_rows(3));
            ui.label("Negative prompt (optional)");ui.text_edit_singleline(&mut self.image_negative);
            egui::Grid::new("image-generation-options").num_columns(4).show(ui,|ui|{
                for (a,b,c,d) in [("Width",&mut self.image_width,"Height",&mut self.image_height),("Steps",&mut self.image_steps,"CFG scale",&mut self.image_cfg),("Seed",&mut self.image_seed,"Vulkan device",&mut self.device_index)] {
                    ui.label(a);ui.text_edit_singleline(b);ui.label(c);ui.text_edit_singleline(d);ui.end_row();
                }
            });
            let path=self.output.clone();self.image_preview(ui,&path);
            return;
        }
        ui.small("Shared image/text packages and native LLaVA, Gemma3, Gemma4 and DeepSeek V4.1. Use images at the package's fixed processor dimensions.");
        path_row(ui,"Tokenizer (optional)",&mut self.tokenizer,true,FileChoice::Either);
        if self.mode==Workflow::VisionTrain {
            path_row(ui,"Image/text JSONL",&mut self.dataset,true,FileChoice::File);
            ui.small("Each row: image path + prompt + completion, or pretokenized input_ids/targets/loss_weights. Image paths are relative to this JSONL file.");
            path_row(ui,"New output directory",&mut self.output,true,FileChoice::Folder);
            egui::Grid::new("vision-training-options").num_columns(4).show(ui,|ui|{
                for (a,b,c,d) in [("Epochs",&mut self.epochs,"Batch size",&mut self.batch_size),("Max sequence length",&mut self.seq_len,"Learning rate",&mut self.learning_rate)] {
                    ui.label(a);ui.text_edit_singleline(b);ui.label(c);ui.text_edit_singleline(d);ui.end_row();
                }
            });
            let native = std::fs::read(Path::new(self.model.trim()).join("config.json"))
                .ok().and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
                .is_some_and(|config| config["model_type"] == "deepseek_v41");
            self.wmma_form(ui, native.then_some("Native DeepSeek V4.1 image training currently supports FP32 precision only."));
        }else{
            path_row(ui,"Input image",&mut self.vision_image,true,FileChoice::File);
            let path=self.vision_image.clone();self.image_preview(ui,&path);
            ui.label("Prompt about this image");ui.add(egui::TextEdit::multiline(&mut self.prompt).desired_width(f32::INFINITY).desired_rows(3));
            egui::Grid::new("vision-generation-options").num_columns(4).show(ui,|ui|{
                ui.label("Max new tokens");ui.text_edit_singleline(&mut self.max_new_tokens);ui.label("Max context");ui.text_edit_singleline(&mut self.seq_len);ui.end_row();
                ui.checkbox(&mut self.do_sample,"Sample");ui.label("");ui.label("");ui.label("");ui.end_row();
                if self.do_sample {ui.label("Temperature");ui.text_edit_singleline(&mut self.temperature);ui.label("Top-k");ui.text_edit_singleline(&mut self.top_k);ui.end_row();ui.label("Top-p");ui.text_edit_singleline(&mut self.top_p);ui.end_row();}
            });
        }
        ui.horizontal(|ui|{ui.label("Vulkan device index");ui.text_edit_singleline(&mut self.device_index);});
        ui.collapsing("Native V4.1 language adapter and image state",|ui|{
            ui.checkbox(&mut self.stream_parameters,"Stream frozen language parameters");
            if self.mode==Workflow::VisionTrain {ui.horizontal(|ui|{ui.label("Language LoRA rank");ui.text_edit_singleline(&mut self.lora_rank);});path_row(ui,"Resume checkpoint (optional)",&mut self.vision_resume,true,FileChoice::Folder);}
            else {path_row(ui,"Trained image export (optional)",&mut self.vision_state,true,FileChoice::Folder);}
            ui.small("V4.1 keeps the selected base model and trains image parameters plus language LoRA. Other families load their complete exported model/ package.");
        });
    }
    fn image_preview(&mut self,ui:&mut egui::Ui,path:&str) {
        if path.trim().is_empty(){return;}
        let key=match std::fs::metadata(path){Ok(m)=>format!("{path}:{}:{:?}",m.len(),m.modified().ok()),Err(_)=>format!("{path}:missing")};
        if key!=self.preview_key {
            self.preview_key=key;self.preview=None;self.preview_error.clear();
            if Path::new(path).is_file(){match image::ImageReader::open(path).and_then(|r|r.with_guessed_format()).map_err(|e|e.to_string()).and_then(|r|r.decode().map_err(|e|e.to_string())) {
                Ok(image)=>{let image=image.thumbnail(320,200).into_rgba8();let size=[image.width() as usize,image.height() as usize];self.preview=Some(ui.ctx().load_texture("multimodal-image-preview",egui::ColorImage::from_rgba_unmultiplied(size,image.as_raw()),egui::TextureOptions::LINEAR));},
                Err(e)=>self.preview_error=format!("Cannot preview image: {e}"),
            }}
        }
        if let Some(texture)=&self.preview {ui.image(texture);}
        if !self.preview_error.is_empty(){ui.small(&self.preview_error);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new(kind:&str)->Self {
            let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let root=env::temp_dir().join(format!("hierarchos-gui-multimodal-{}-{nonce}",std::process::id()));
            std::fs::create_dir_all(root.join("model")).unwrap();
            std::fs::write(root.join("model/config.json"),serde_json::to_vec(&serde_json::json!({"model_type":kind})).unwrap()).unwrap();
            for name in ["image.png","dataset.jsonl","checkpoint.gguf"]{std::fs::write(root.join(name),b"fixture").unwrap();}
            Self(root)
        }
        fn app(&self,mode:Workflow)->NativeApp {
            let mut app=NativeApp::default();app.mode=mode;app.model=self.0.join("model").display().to_string();app.vision_image=self.0.join("image.png").display().to_string();
            app.dataset=self.0.join("dataset.jsonl").display().to_string();app.output=self.0.join("new-output").display().to_string();app.prompt="Describe this image".into();app.batch_size="1".into();app
        }
    }
    impl Drop for Fixture {fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
    #[test]
    fn vision_workflows_build_real_image_commands_without_text_cache_flags() {
        let f=Fixture::new("hierarchos_image_text");let mut app=f.app(Workflow::VisionGenerate);
        app.continuous_batching=true;app.generation_cache_mode=GenerationCacheMode::Paged;app.do_sample=false;
        let args=app.build_args().unwrap();assert_eq!(args[0],"vision-generate");
        assert!(args.windows(2).any(|a|a[0]=="--image"&&a[1]==app.vision_image));
        assert!(args.iter().any(|a|a=="--max-length"));assert!(!args.iter().any(|a|a.contains("cache")||a.contains("continuous")));
        app.mode=Workflow::VisionTrain;let args=app.build_args().unwrap();assert_eq!(args[0],"vision-train");
        assert!(args.windows(2).any(|a|a[0]=="--train"&&a[1]==app.dataset));assert!(!args.iter().any(|a|a=="--full-finetune"));
    }
    #[test]
    fn vision_precision_reaches_each_supported_backend_and_rejects_unimplemented_modes() {
        for kind in ["hierarchos_image_text", "llava", "gemma3", "gemma4"] {
            let fixture = Fixture::new(kind);
            let mut app = fixture.app(Workflow::VisionTrain);
            for (choice, flag) in [(WmmaModeChoice::Off, "off"), (WmmaModeChoice::Mixed, "mixed")] {
                app.wmma_mode = choice;
                let args = app.build_args().unwrap();
                assert_eq!(args.iter().filter(|a| *a == "--wmma-mode").count(), 1);
                assert!(args.windows(2).any(|a| a == ["--wmma-mode", flag]), "{kind}: {args:?}");
            }
            app.wmma_mode = WmmaModeChoice::FullFp16;
            assert!(app.build_args().unwrap_err().contains("unavailable"));
            app.mode = Workflow::VisionGenerate;
            assert!(!app.build_args().unwrap().iter().any(|a| a == "--wmma-mode"));
        }
        let fixture = Fixture::new("deepseek_v41");
        let mut app = fixture.app(Workflow::VisionTrain);
        assert!(app.build_args().unwrap().windows(2).any(|a| a == ["--wmma-mode", "off"]));
        app.wmma_mode = WmmaModeChoice::Mixed;
        assert!(app.build_args().unwrap_err().contains("currently uses FP32"));
    }

    #[test]
    fn text_to_image_uses_image_model_file_output_and_generation_controls() {
        let f=Fixture::new("llava");let mut app=f.app(Workflow::ImageGenerate);
        app.model=f.0.join("checkpoint.gguf").display().to_string();app.output=f.0.join("generated.png").display().to_string();app.image_negative="blur".into();
        let args=app.build_args().unwrap();assert_eq!(args[0],"image-generate");
        for flag in ["--output","--width","--height","--steps","--cfg-scale","--seed","--negative-prompt"]{assert!(args.iter().any(|a|a==flag),"{flag}");}
        assert!(!args.iter().any(|a|a=="--image"||a=="--max-new-tokens"||a=="--train"));
        app.image_width="513".into();assert!(app.build_args().is_err());app.image_width="512".into();
        std::fs::write(&app.output,b"existing image").unwrap();assert!(app.build_args().is_err());
    }
    #[test]
    fn native_image_state_is_not_silently_sent_to_other_architectures() {
        let f=Fixture::new("gemma3");let mut app=f.app(Workflow::VisionGenerate);
        app.vision_state=f.0.join("model").display().to_string();assert!(app.build_args().unwrap_err().contains("V4.1"));
        let f=Fixture::new("deepseek_v41");let mut app=f.app(Workflow::VisionTrain);app.stream_parameters=true;
        let args=app.build_args().unwrap();assert!(args.iter().any(|a|a=="--lora-rank"));assert!(args.iter().any(|a|a=="--stream-parameters"));
        app.batch_size="2".into();assert!(app.build_args().is_err());
    }
    #[test]
    fn vision_training_preserves_existing_output_and_accepts_empty_destination() {
        let f=Fixture::new("hierarchos_image_text");let app=f.app(Workflow::VisionTrain);
        std::fs::create_dir(&app.output).unwrap();assert!(app.build_args().is_ok());
        std::fs::write(Path::new(&app.output).join("weights"),b"existing").unwrap();assert!(app.build_args().is_err());
    }
    #[test]
    #[ignore = "requires HIERARCHOS_MIXED_GUI_FIXTURE, HIERARCHOS_NATIVE_CLI and a cooperative-matrix GPU"]
    fn mixed_training_start_executes_cooperative_workers() {
        let root = PathBuf::from(env::var_os("HIERARCHOS_MIXED_GUI_FIXTURE").expect("eligible fixture root"));
        let mut app = NativeApp::default();
        app.wmma_mode = WmmaModeChoice::Mixed;
        app.cli_path = env::var("HIERARCHOS_NATIVE_CLI").expect("fresh native CLI");
        app.dataset = root.join("dataset.jsonl").display().to_string();
        app.epochs = "2".into();
        app.batch_size = "2".into();
        app.seq_len = "24".into();
        app.learning_rate = ".001".into();
        for (mode, name) in [(Workflow::VisionTrain, "vision"), (Workflow::TransformerTrain, "full"),
            (Workflow::TransformerFullFinetune, "full-finetune"), (Workflow::TransformerFinetune, "lora")] {
            app.mode = mode;
            app.model = root.join(if mode == Workflow::VisionTrain { "model" } else { "model/text" }).display().to_string();
            app.output = root.join(format!("gui-mixed-{name}")).display().to_string();
            app.start();
            assert!(app.child.is_some(), "{}", app.status);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
            while app.child.is_some() {
                app.drain_output();
                if std::time::Instant::now() > deadline {
                    app.stop();
                    panic!("mixed vision worker timed out: {}", app.log);
                }
                thread::sleep(std::time::Duration::from_millis(20));
            }
            assert_eq!(app.status, "Finished successfully", "{}", app.log);
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(Path::new(&app.output).join("training_report.json")).unwrap()
            ).unwrap();
            assert_eq!(report["steps"], 2);
            assert_eq!(report["precision"]["requested"], "mixed");
            for phase in ["cooperative_dispatches", "forward", "input_grad"] {
                assert!(report["precision"][phase].as_u64().unwrap() > 0, "{phase}: {report}");
            }
            assert_eq!(report["precision"]["weight_grad"].as_u64().unwrap() > 0, mode != Workflow::TransformerFinetune);
        }
    }

    #[test]
    #[ignore = "requires HIERARCHOS_MULTIMODAL_GUI_SMOKE manifest and local Vulkan image models"]
    fn multimodal_start_runs_native_vision_and_image_generation_workers() {
        let path=env::var_os("HIERARCHOS_MULTIMODAL_GUI_SMOKE").expect("GUI smoke manifest");
        let v:serde_json::Value=serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let get=|name:&str|v[name].as_str().unwrap().to_owned();
        let mut app=NativeApp::default();app.cli_path=get("cli");app.model=get("model");app.vision_image=get("image");app.tokenizer=get("tokenizer");
        app.dataset=get("dataset");app.prompt="t3 t4".into();app.seq_len="8".into();app.max_new_tokens="1".into();app.do_sample=false;
        app.epochs="1".into();app.batch_size="2".into();app.learning_rate="0.0001".into();
        fn run(app:&mut NativeApp) {
            app.start();assert!(app.child.is_some(),"{}",app.status);
            let deadline=std::time::Instant::now()+std::time::Duration::from_secs(1200);
            let mut printed=0;
            while app.child.is_some(){app.drain_output();if app.log.len()>printed{eprint!("{}",&app.log[printed..]);printed=app.log.len();}if std::time::Instant::now()>deadline{app.stop();panic!("worker timeout: {}",app.log);}thread::sleep(std::time::Duration::from_millis(50));}
            assert_eq!(app.status,"Finished successfully","{}",app.log);
            eprintln!("GUI worker {}:\n{}",app.mode.cli_mode(),app.log);
        }
        app.mode=Workflow::VisionTrain;app.output=get("training_output");run(&mut app);
        assert!(Path::new(&app.output).join("model/config.json").is_file());
        app.model=Path::new(&app.output).join("model").display().to_string();app.mode=Workflow::VisionGenerate;run(&mut app);
        assert!(app.log.contains("Vision backend"));
        app.mode=Workflow::ImageGenerate;app.model=get("image_model");app.image_engine=get("engine");app.output=get("generated_png");
        app.prompt="a red cube on a white background".into();app.image_width="64".into();app.image_height="64".into();app.image_steps="1".into();app.image_cfg="1".into();run(&mut app);
        let picture=image::open(&app.output).unwrap();assert_eq!((picture.width(),picture.height()),(64,64));
        let receipt:serde_json::Value=serde_json::from_slice(&std::fs::read(Path::new(&app.output).with_extension("generation.json")).unwrap()).unwrap();
        assert_eq!(receipt["backend"],"stable-diffusion.cpp-vulkan");assert_eq!(receipt["automatic_offload"],false);
    }
}
