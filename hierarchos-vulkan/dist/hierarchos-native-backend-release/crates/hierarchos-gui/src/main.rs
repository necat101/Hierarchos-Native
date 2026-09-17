#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::{
    env,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
};

use eframe::egui;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Workflow {
    TransformerTrain,
    TransformerFinetune,
    HierarchosTrain,
    HierarchosFinetune,
    TransformerGenerate,
    HierarchosInference,
}

impl Workflow {
    fn cli_mode(self) -> &'static str {
        match self {
            Self::TransformerTrain => "transformer-train",
            Self::TransformerFinetune => "transformer-finetune",
            Self::HierarchosTrain => "train",
            Self::HierarchosFinetune => "finetune",
            Self::TransformerGenerate => "transformer-generate",
            Self::HierarchosInference => "chat",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::TransformerTrain => "Transformer full training",
            Self::TransformerFinetune => "Transformer LoRA fine-tuning",
            Self::HierarchosTrain => "Hierarchos training",
            Self::HierarchosFinetune => "Hierarchos fine-tuning",
            Self::TransformerGenerate => "Transformer inference / generation",
            Self::HierarchosInference => "Hierarchos inference / chat",
        }
    }

    fn is_transformer(self) -> bool {
        matches!(
            self,
            Self::TransformerTrain | Self::TransformerFinetune | Self::TransformerGenerate
        )
    }

    fn is_training(self) -> bool {
        matches!(
            self,
            Self::TransformerTrain
                | Self::TransformerFinetune
                | Self::HierarchosTrain
                | Self::HierarchosFinetune
        )
    }

    fn action_label(self) -> &'static str {
        if self.is_training() {
            "Start training"
        } else {
            "Run inference"
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationCacheMode {
    ModelDefault,
    Disabled,
    Contiguous,
    Paged,
}

impl GenerationCacheMode {
    fn label(self) -> &'static str {
        match self {
            Self::ModelDefault => "Model/package default (non-paged)",
            Self::Disabled => "Disabled (full-prefix)",
            Self::Contiguous => "Contiguous native KV",
            Self::Paged => "Paged Vulkan KV",
        }
    }
}

struct NativeApp {
    mode: Workflow,
    hf_model: bool,
    hf_dataset: bool,
    model: String,
    tokenizer: String,
    dataset: String,
    output: String,
    epochs: String,
    batch_size: String,
    seq_len: String,
    learning_rate: String,
    device_index: String,
    lora_rank: String,
    assistant_recovery: bool,
    prompt: String,
    max_new_tokens: String,
    temperature: String,
    top_k: String,
    top_p: String,
    do_sample: bool,
    generation_cache_mode: GenerationCacheMode,
    continuous_batching: bool,
    continuous_batch_max_requests: String,
    continuous_batch_prompts: String,
    cli_path: String,
    child: Option<Child>,
    output_rx: Option<Receiver<String>>,
    log: String,
    status: String,
    last_command: String,
}

impl Default for NativeApp {
    fn default() -> Self {
        Self {
            mode: Workflow::TransformerFinetune,
            hf_model: false,
            hf_dataset: false,
            model: String::new(),
            tokenizer: String::new(),
            dataset: String::new(),
            output: String::new(),
            epochs: "3".to_owned(),
            batch_size: "4".to_owned(),
            seq_len: "512".to_owned(),
            learning_rate: "5e-5".to_owned(),
            device_index: String::new(),
            lora_rank: "8".to_owned(),
            assistant_recovery: false,
            prompt: String::new(),
            max_new_tokens: "256".to_owned(),
            temperature: "0.7".to_owned(),
            top_k: "50".to_owned(),
            top_p: "0.9".to_owned(),
            do_sample: true,
            generation_cache_mode: GenerationCacheMode::ModelDefault,
            continuous_batching: false,
            continuous_batch_max_requests: String::new(),
            continuous_batch_prompts: String::new(),
            cli_path: discover_cli().display().to_string(),
            child: None,
            output_rx: None,
            log: String::new(),
            status: "Ready".to_owned(),
            last_command: String::new(),
        }
    }
}

impl Drop for NativeApp {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl NativeApp {
    fn drain_output(&mut self) {
        if let Some(rx) = self.output_rx.as_ref() {
            while let Ok(line) = rx.try_recv() {
                self.log.push_str(&line);
                self.log.push('\n');
            }
        }
        if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    self.status = if status.success() {
                        "Finished successfully".to_owned()
                    } else {
                        format!("Exited with {status}")
                    };
                    self.child = None;
                    self.output_rx = None;
                }
                Ok(None) => {}
                Err(error) => {
                    self.status = format!("Could not query worker: {error}");
                    self.child = None;
                    self.output_rx = None;
                }
            }
        }
    }

    fn build_args(&self) -> Result<Vec<String>, String> {
        if self.mode.is_training() {
            self.build_training_args()
        } else {
            self.build_inference_args()
        }
    }

    fn build_training_args(&self) -> Result<Vec<String>, String> {
        let mut args = vec![self.mode.cli_mode().to_owned()];

        let model = self.model.trim();
        if !model.is_empty() {
            if self.hf_model {
                args.extend(["--hf-model".to_owned(), model.to_owned()]);
            } else {
                require_directory(model, "model directory")?;
                args.extend(["--model-path".to_owned(), model.to_owned()]);
            }
        } else if self.mode != Workflow::HierarchosTrain {
            return Err("Choose a model directory or Hugging Face model ID.".to_owned());
        }

        let tokenizer = self.tokenizer.trim();
        if !tokenizer.is_empty() {
            if looks_like_hf_repo(tokenizer) && !Path::new(tokenizer).exists() {
                args.extend(["--hf-tokenizer".to_owned(), tokenizer.to_owned()]);
            } else {
                require_existing(tokenizer, "tokenizer path")?;
                args.extend(["--tokenizer-path".to_owned(), tokenizer.to_owned()]);
            }
        } else if self.mode == Workflow::HierarchosTrain && model.is_empty() {
            return Err(
                "Fresh Hierarchos training needs a tokenizer directory/file or Hugging Face tokenizer ID."
                    .to_owned(),
            );
        }

        let dataset = self.dataset.trim();
        if dataset.is_empty() {
            return Err("Choose a training dataset or Hugging Face dataset ID.".to_owned());
        }
        if self.hf_dataset {
            args.extend(["--hf-dataset".to_owned(), dataset.to_owned()]);
        } else {
            if self.mode.is_transformer() {
                require_file(dataset, "Transformer dataset file")?;
            } else {
                require_existing(dataset, "dataset path")?;
            }
            args.extend(["--train".to_owned(), dataset.to_owned()]);
        }

        let output = self.output.trim();
        if output.is_empty() {
            return Err("Choose an output directory.".to_owned());
        }
        args.extend(["--out-dir".to_owned(), output.to_owned()]);

        push_positive(&mut args, "--epochs", &self.epochs)?;
        push_positive(&mut args, "--batch-size", &self.batch_size)?;
        push_positive(&mut args, "--max-length", &self.seq_len)?;
        push_positive_float(&mut args, "--lr", &self.learning_rate)?;
        if !self.device_index.trim().is_empty() {
            self.device_index
                .trim()
                .parse::<usize>()
                .map_err(|_| "Device index must be a non-negative integer.".to_owned())?;
            args.extend([
                "--device-index".to_owned(),
                self.device_index.trim().to_owned(),
            ]);
        }
        if self.mode == Workflow::TransformerFinetune && !self.lora_rank.trim().is_empty() {
            push_positive(&mut args, "--lora-rank", &self.lora_rank)?;
        }
        if self.mode == Workflow::HierarchosTrain && self.assistant_recovery {
            args.push("--assistant-recovery".to_owned());
        }
        Ok(args)
    }

    fn build_inference_args(&self) -> Result<Vec<String>, String> {
        let mut args = vec![self.mode.cli_mode().to_owned()];
        let model = self.model.trim();
        if model.is_empty() {
            return Err("Choose a model directory or Hugging Face model ID.".to_owned());
        }
        if self.mode == Workflow::HierarchosInference && self.hf_model {
            return Err(
                "Hierarchos inference requires a local native model package; pull or convert the model first."
                    .to_owned(),
            );
        }
        if self.hf_model {
            args.extend(["--hf-model".to_owned(), model.to_owned()]);
        } else {
            require_directory(model, "model directory")?;
            args.extend(["--model-path".to_owned(), model.to_owned()]);
        }

        let prompt = self.prompt.trim();
        if prompt.is_empty() {
            return Err("Enter a prompt for inference.".to_owned());
        }
        args.extend(["--prompt".to_owned(), prompt.to_owned()]);
        push_positive(&mut args, "--max-new-tokens", &self.max_new_tokens)?;

        let temperature = parse_positive_float("--temperature", &self.temperature)?;
        let top_k = self
            .top_k
            .trim()
            .parse::<usize>()
            .map_err(|_| "--top-k must be a non-negative integer.".to_owned())?;
        let top_p = parse_probability("--top-p", &self.top_p)?;

        match self.mode {
            Workflow::TransformerGenerate => {
                if !self.device_index.trim().is_empty() {
                    self.device_index
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| "Device index must be a non-negative integer.".to_owned())?;
                    args.extend([
                        "--device-index".to_owned(),
                        self.device_index.trim().to_owned(),
                    ]);
                }
                if self.do_sample {
                    args.push("--do-sample".to_owned());
                    args.extend(["--temperature".to_owned(), temperature.to_string()]);
                    args.extend(["--top-k".to_owned(), top_k.to_string()]);
                    args.extend(["--top-p".to_owned(), top_p.to_string()]);
                } else {
                    args.push("--no-do-sample".to_owned());
                }
                match self.generation_cache_mode {
                    GenerationCacheMode::ModelDefault => {}
                    GenerationCacheMode::Disabled => args.push("--no-use-cache".to_owned()),
                    GenerationCacheMode::Contiguous => {
                        args.push("--use-cache".to_owned());
                        args.extend(["--cache-implementation".to_owned(), "dynamic".to_owned()]);
                    }
                    GenerationCacheMode::Paged => {
                        args.push("--use-cache".to_owned());
                        args.extend(["--cache-implementation".to_owned(), "paged".to_owned()]);
                    }
                }
                if self.continuous_batching {
                    if self.generation_cache_mode != GenerationCacheMode::Paged {
                        return Err(
                            "Continuous batching requires Paged Vulkan KV; choose Paged Vulkan KV explicitly."
                                .to_owned(),
                        );
                    }
                    args.push("--continuous-batching".to_owned());
                    let max_requests = self.continuous_batch_max_requests.trim();
                    if !max_requests.is_empty() {
                        let parsed = max_requests.parse::<usize>().map_err(|_| {
                            "Continuous batch max requests must be a positive integer.".to_owned()
                        })?;
                        if parsed == 0 {
                            return Err(
                                "Continuous batch max requests must be at least 1.".to_owned()
                            );
                        }
                        args.extend([
                            "--continuous-batch-max-requests".to_owned(),
                            parsed.to_string(),
                        ]);
                    }
                    for additional_prompt in self
                        .continuous_batch_prompts
                        .lines()
                        .map(str::trim)
                        .filter(|prompt| !prompt.is_empty())
                    {
                        args.extend(["--prompt".to_owned(), additional_prompt.to_owned()]);
                    }
                }
            }
            Workflow::HierarchosInference => {
                args.extend(["--temperature".to_owned(), temperature.to_string()]);
                args.extend(["--top-k".to_owned(), top_k.to_string()]);
                args.extend(["--top-p".to_owned(), top_p.to_string()]);
            }
            _ => unreachable!("training workflows are handled by build_training_args"),
        }
        Ok(args)
    }

    fn start(&mut self) {
        if self.child.is_some() {
            return;
        }
        let args = match self.build_args() {
            Ok(args) => args,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        let cli = self.cli_path.trim();
        if cli.is_empty() {
            self.status = "CLI path is empty.".to_owned();
            return;
        }
        self.last_command = command_preview(cli, &args);
        self.log.clear();
        self.log.push_str(&format!("$ {}\n", self.last_command));

        let mut command = Command::new(cli);
        command
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.status = format!("Could not start native CLI: {error}");
                return;
            }
        };

        let (tx, rx) = mpsc::channel::<String>();
        if let Some(stdout) = child.stdout.take() {
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    let _ = tx.send(line);
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    let _ = tx.send(line);
                }
            });
        }
        self.output_rx = Some(rx);
        self.child = Some(child);
        self.status = if self.mode.is_training() {
            "Training is running".to_owned()
        } else {
            "Inference is running".to_owned()
        };
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            match child.kill() {
                Ok(()) => {
                    let _ = child.wait();
                    self.status = "Operation stopped".to_owned();
                    self.log.push_str("[GUI] Native worker stopped.\n");
                }
                Err(error) => self.status = format!("Could not stop worker: {error}"),
            }
        }
        self.output_rx = None;
    }

    fn run_utility(&mut self, subcommand: &str) {
        if self.child.is_some() {
            self.status = "Stop the running operation before using a diagnostic.".to_owned();
            return;
        }
        let cli = self.cli_path.trim();
        match Command::new(cli).arg(subcommand).output() {
            Ok(output) => {
                self.log.clear();
                self.log.push_str(&String::from_utf8_lossy(&output.stdout));
                self.log.push_str(&String::from_utf8_lossy(&output.stderr));
                self.status = if output.status.success() {
                    format!("{subcommand} completed")
                } else {
                    format!("{subcommand} exited with {}", output.status)
                };
            }
            Err(error) => self.status = format!("Could not run {subcommand}: {error}"),
        }
    }
}

impl eframe::App for NativeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_output();
        if self.child.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.heading("Hierarchos Native");
            ui.label(
                "Native Transformer + Hierarchos training and inference without a Python runtime",
            );
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.monospace(&self.status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Workflow");
                egui::ComboBox::from_id_salt("workflow")
                    .selected_text(self.mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [
                            Workflow::TransformerTrain,
                            Workflow::TransformerFinetune,
                            Workflow::TransformerGenerate,
                            Workflow::HierarchosTrain,
                            Workflow::HierarchosFinetune,
                            Workflow::HierarchosInference,
                        ] {
                            ui.selectable_value(&mut self.mode, mode, mode.label());
                        }
                    });
            });
            ui.add_space(6.0);

            if self.mode == Workflow::HierarchosInference {
                self.hf_model = false;
            }

            path_row(
                ui,
                "Model",
                &mut self.model,
                !self.hf_model,
                FileChoice::Folder,
            );
            ui.horizontal(|ui| {
                ui.add_enabled(
                    self.mode != Workflow::HierarchosInference,
                    egui::Checkbox::new(
                        &mut self.hf_model,
                        "Model is a Hugging Face OWNER/REPO ID",
                    ),
                );
                if self.mode == Workflow::HierarchosTrain {
                    ui.label("(optional for fresh Hierarchos initialization)");
                } else if self.mode == Workflow::HierarchosInference {
                    ui.label("(native Hierarchos inference uses a local package)");
                }
            });

            if self.mode.is_training() {
                path_row(
                    ui,
                    "Tokenizer (optional)",
                    &mut self.tokenizer,
                    true,
                    FileChoice::Either,
                );
                path_row(
                    ui,
                    "Training data",
                    &mut self.dataset,
                    !self.hf_dataset,
                    if self.mode.is_transformer() {
                        FileChoice::File
                    } else {
                        FileChoice::Either
                    },
                );
                ui.checkbox(
                    &mut self.hf_dataset,
                    "Dataset is a Hugging Face OWNER/REPO ID",
                );
                path_row(ui, "Output", &mut self.output, true, FileChoice::Folder);

                ui.separator();
                egui::Grid::new("training-options")
                    .num_columns(4)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Epochs");
                        ui.text_edit_singleline(&mut self.epochs);
                        ui.label("Batch size");
                        ui.text_edit_singleline(&mut self.batch_size);
                        ui.end_row();
                        ui.label("Max sequence length");
                        ui.text_edit_singleline(&mut self.seq_len);
                        ui.label("Learning rate");
                        ui.text_edit_singleline(&mut self.learning_rate);
                        ui.end_row();
                        ui.label("Vulkan device index");
                        ui.text_edit_singleline(&mut self.device_index);
                        if self.mode == Workflow::TransformerFinetune {
                            ui.label("LoRA rank");
                            ui.text_edit_singleline(&mut self.lora_rank);
                        } else {
                            ui.label("");
                            ui.label("");
                        }
                        ui.end_row();
                    });
                if self.mode == Workflow::HierarchosTrain {
                    ui.checkbox(
                        &mut self.assistant_recovery,
                        "Assistant-recovery SFT preset (Alpaca + response-focused defaults)",
                    );
                }
            } else {
                ui.separator();
                ui.label("Prompt");
                ui.add(
                    egui::TextEdit::multiline(&mut self.prompt)
                        .desired_width(f32::INFINITY)
                        .desired_rows(5),
                );
                egui::Grid::new("inference-options")
                    .num_columns(4)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Max new tokens");
                        ui.text_edit_singleline(&mut self.max_new_tokens);
                        ui.label("Temperature");
                        ui.text_edit_singleline(&mut self.temperature);
                        ui.end_row();
                        ui.label("Top-k");
                        ui.text_edit_singleline(&mut self.top_k);
                        ui.label("Top-p");
                        ui.text_edit_singleline(&mut self.top_p);
                        ui.end_row();
                        if self.mode == Workflow::TransformerGenerate {
                            ui.label("Vulkan device index");
                            ui.text_edit_singleline(&mut self.device_index);
                            ui.checkbox(&mut self.do_sample, "Sample");
                            ui.label(if self.do_sample {
                                "temperature/top-k/top-p enabled"
                            } else {
                                "greedy/beam policy from generation config"
                            });
                            ui.end_row();
                            ui.label("KV cache");
                            let cache_changed = egui::ComboBox::from_id_salt("generation-kv-cache")
                                .selected_text(self.generation_cache_mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in [
                                        GenerationCacheMode::ModelDefault,
                                        GenerationCacheMode::Disabled,
                                        GenerationCacheMode::Contiguous,
                                        GenerationCacheMode::Paged,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.generation_cache_mode,
                                            mode,
                                            mode.label(),
                                        );
                                    }
                                })
                                .response
                                .changed();
                            if cache_changed
                                && self.generation_cache_mode != GenerationCacheMode::Paged
                            {
                                self.continuous_batching = false;
                            }
                            ui.label("Paged is opt-in; package metadata cannot enable it implicitly");
                            ui.label("");
                            ui.end_row();
                            ui.label("Continuous batching");
                            let changed = ui
                                .checkbox(
                                    &mut self.continuous_batching,
                                    "Share paged KV arenas across active requests",
                                )
                                .changed();
                            if changed && self.continuous_batching {
                                self.generation_cache_mode = GenerationCacheMode::Paged;
                            }
                            ui.label("Opt-in; implies Paged Vulkan KV");
                            ui.label("");
                            ui.end_row();
                            if self.continuous_batching {
                                ui.label("Max active requests");
                                ui.text_edit_singleline(&mut self.continuous_batch_max_requests);
                                ui.label("Blank = all supplied requests");
                                ui.label("");
                                ui.end_row();
                            }
                        }
                    });
                if self.mode == Workflow::TransformerGenerate && self.continuous_batching {
                    ui.label("Additional continuous-batch prompts (one prompt per line)");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.continuous_batch_prompts)
                            .desired_width(f32::INFINITY)
                            .desired_rows(3),
                    );
                    ui.small(
                        "The main Prompt is request 1. Each non-empty line above becomes another --prompt request. Finished requests release paged KV blocks for reuse by waiting requests.",
                    );
                }
                if self.mode == Workflow::TransformerGenerate {
                    ui.small(
                        "Transformer inference honors generation_config.json except package-level paged/continuous serving opt-ins; select them here explicitly (or pass an explicit generation config in the CLI).",
                    );
                } else {
                    ui.small(
                        "Hierarchos inference uses the pure-Rust runtime and the tokenizer bundled with the model package.",
                    );
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Native CLI");
                ui.text_edit_singleline(&mut self.cli_path);
                if ui.button("Auto-detect").clicked() {
                    self.cli_path = discover_cli().display().to_string();
                }
            });
            ui.horizontal(|ui| {
                let running = self.child.is_some();
                if ui
                    .add_enabled(!running, egui::Button::new(self.mode.action_label()))
                    .clicked()
                {
                    self.start();
                }
                if ui.add_enabled(running, egui::Button::new("Stop")).clicked() {
                    self.stop();
                }
                if ui
                    .add_enabled(!running, egui::Button::new("Check backend"))
                    .clicked()
                {
                    self.run_utility("doctor");
                }
                if ui
                    .add_enabled(!running, egui::Button::new("List architectures"))
                    .clicked()
                {
                    self.run_utility("architectures");
                }
            });

            if let Ok(args) = self.build_args() {
                ui.collapsing("Command preview", |ui| {
                    ui.monospace(command_preview(self.cli_path.trim(), &args));
                });
            }

            ui.separator();
            ui.label("Output");
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .max_height(320.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.log)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(14)
                            .interactive(false),
                    );
                });
        });
    }
}

#[derive(Clone, Copy)]
enum FileChoice {
    File,
    Folder,
    Either,
}

fn path_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    browse_enabled: bool,
    choice: FileChoice,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(value).desired_width(520.0));
        let picked = match choice {
            FileChoice::File => ui
                .add_enabled(browse_enabled, egui::Button::new("Browse file…"))
                .clicked()
                .then(|| rfd::FileDialog::new().pick_file())
                .flatten(),
            FileChoice::Folder => ui
                .add_enabled(browse_enabled, egui::Button::new("Browse folder…"))
                .clicked()
                .then(|| rfd::FileDialog::new().pick_folder())
                .flatten(),
            FileChoice::Either => {
                let file = ui
                    .add_enabled(browse_enabled, egui::Button::new("File…"))
                    .clicked()
                    .then(|| rfd::FileDialog::new().pick_file())
                    .flatten();
                if file.is_some() {
                    file
                } else {
                    ui.add_enabled(browse_enabled, egui::Button::new("Folder…"))
                        .clicked()
                        .then(|| rfd::FileDialog::new().pick_folder())
                        .flatten()
                }
            }
        };
        if let Some(path) = picked {
            *value = path.display().to_string();
        }
    });
}

fn discover_cli() -> PathBuf {
    if let Some(path) = env::var_os("HIERARCHOS_NATIVE_CLI") {
        return PathBuf::from(path);
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(platform_executable("hierarchos-native-cli"));
            if sibling.is_file() {
                return sibling;
            }
            let bin_sibling = dir
                .join("bin")
                .join(platform_executable("hierarchos-native-cli"));
            if bin_sibling.is_file() {
                return bin_sibling;
            }
        }
    }
    PathBuf::from(platform_executable("hierarchos-native-cli"))
}

fn platform_executable(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

fn require_existing(value: &str, label: &str) -> Result<(), String> {
    if Path::new(value).exists() {
        Ok(())
    } else {
        Err(format!("The {label} does not exist: {value}"))
    }
}

fn require_directory(value: &str, label: &str) -> Result<(), String> {
    if Path::new(value).is_dir() {
        Ok(())
    } else {
        Err(format!("The {label} is not a directory: {value}"))
    }
}

fn require_file(value: &str, label: &str) -> Result<(), String> {
    if Path::new(value).is_file() {
        Ok(())
    } else {
        Err(format!("The {label} is not a file: {value}"))
    }
}

fn looks_like_hf_repo(value: &str) -> bool {
    let mut parts = value.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(repo), None) if !owner.is_empty() && !repo.is_empty())
}

fn push_positive(args: &mut Vec<String>, option: &str, raw: &str) -> Result<(), String> {
    let value = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| format!("{option} must be a positive integer."))?;
    if value == 0 {
        return Err(format!("{option} must be positive."));
    }
    args.extend([option.to_owned(), value.to_string()]);
    Ok(())
}

fn push_positive_float(args: &mut Vec<String>, option: &str, raw: &str) -> Result<(), String> {
    let value = parse_positive_float(option, raw)?;
    args.extend([option.to_owned(), value.to_string()]);
    Ok(())
}

fn parse_positive_float(option: &str, raw: &str) -> Result<f64, String> {
    let value = raw
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("{option} must be a positive number."))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{option} must be a finite positive number."));
    }
    Ok(value)
}

fn parse_probability(option: &str, raw: &str) -> Result<f64, String> {
    let value = parse_positive_float(option, raw)?;
    if value > 1.0 {
        return Err(format!("{option} must be greater than 0 and at most 1."));
    }
    Ok(value)
}

fn command_preview(cli: &str, args: &[String]) -> String {
    std::iter::once(cli)
        .chain(args.iter().map(String::as_str))
        .map(quote_preview)
        .collect::<Vec<_>>()
        .join(" ")
}

fn quote_preview(value: &str) -> String {
    if value.chars().all(|c| !c.is_whitespace() && c != '"') {
        value.to_owned()
    } else {
        format!("\"{}\"", value.replace('"', "\\\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_hf_inference_builds_generation_command() {
        let mut app = NativeApp::default();
        app.mode = Workflow::TransformerGenerate;
        app.hf_model = true;
        app.model = "openai-community/gpt2".to_owned();
        app.prompt = "Hello native Vulkan".to_owned();
        app.max_new_tokens = "32".to_owned();
        app.device_index = "1".to_owned();
        app.do_sample = false;
        app.temperature = "0.8".to_owned();
        app.generation_cache_mode = GenerationCacheMode::Paged;
        let args = app.build_args().expect("valid Transformer generation args");
        assert_eq!(args[0], "transformer-generate");
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--hf-model", "openai-community/gpt2"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--prompt", "Hello native Vulkan"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--max-new-tokens", "32"]));
        assert!(args.windows(2).any(|pair| pair == ["--device-index", "1"]));
        assert!(args.iter().any(|arg| arg == "--no-do-sample"));
        assert!(!args.iter().any(|arg| arg == "--temperature"));
        assert!(args.iter().any(|arg| arg == "--use-cache"));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--cache-implementation", "paged"]));
    }

    #[test]
    fn transformer_generation_cache_selector_can_force_contiguous_or_disable_cache() {
        let mut app = NativeApp::default();
        app.mode = Workflow::TransformerGenerate;
        app.hf_model = true;
        app.model = "owner/model".to_owned();
        app.prompt = "Hello".to_owned();
        app.do_sample = false;

        app.generation_cache_mode = GenerationCacheMode::Contiguous;
        let contiguous = app.build_args().expect("contiguous cache args");
        assert!(contiguous.iter().any(|arg| arg == "--use-cache"));
        assert!(contiguous
            .windows(2)
            .any(|pair| pair == ["--cache-implementation", "dynamic"]));

        app.generation_cache_mode = GenerationCacheMode::Disabled;
        let disabled = app.build_args().expect("disabled cache args");
        assert!(disabled.iter().any(|arg| arg == "--no-use-cache"));
        assert!(!disabled.iter().any(|arg| arg == "--cache-implementation"));
    }

    #[test]
    fn transformer_generation_continuous_batching_is_explicit_and_repeats_prompts() {
        let mut app = NativeApp::default();
        app.mode = Workflow::TransformerGenerate;
        app.hf_model = true;
        app.model = "owner/model".to_owned();
        app.prompt = "primary prompt\nmay be multiline".to_owned();
        app.do_sample = false;
        app.generation_cache_mode = GenerationCacheMode::Paged;
        app.continuous_batching = true;
        app.continuous_batch_max_requests = "2".to_owned();
        app.continuous_batch_prompts = "second request\n\nthird request".to_owned();

        let args = app.build_args().expect("continuous batch generation args");
        assert!(args.iter().any(|arg| arg == "--continuous-batching"));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--continuous-batch-max-requests", "2"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--cache-implementation", "paged"]));
        let prompts = args
            .windows(2)
            .filter(|pair| pair[0] == "--prompt")
            .map(|pair| pair[1].as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            prompts,
            vec![
                "primary prompt\nmay be multiline",
                "second request",
                "third request"
            ]
        );
    }

    #[test]
    fn transformer_generation_continuous_batching_rejects_nonpaged_cache_modes() {
        let mut app = NativeApp::default();
        app.mode = Workflow::TransformerGenerate;
        app.hf_model = true;
        app.model = "owner/model".to_owned();
        app.prompt = "Hello".to_owned();
        app.do_sample = false;
        app.continuous_batching = true;

        for mode in [
            GenerationCacheMode::ModelDefault,
            GenerationCacheMode::Disabled,
            GenerationCacheMode::Contiguous,
        ] {
            app.generation_cache_mode = mode;
            let error = app
                .build_args()
                .expect_err("continuous batching must reject non-paged cache modes");
            assert!(error.contains("requires Paged Vulkan KV"));
        }
    }

    #[test]
    fn hierarchos_inference_refuses_remote_model_ids() {
        let mut app = NativeApp::default();
        app.mode = Workflow::HierarchosInference;
        app.hf_model = true;
        app.model = "owner/model".to_owned();
        app.prompt = "Hello".to_owned();
        let error = app
            .build_args()
            .expect_err("remote Hierarchos inference must fail closed");
        assert!(error.contains("local native model package"));
    }

    #[test]
    fn inference_requires_prompt_and_probability_is_bounded() {
        let mut app = NativeApp::default();
        app.mode = Workflow::TransformerGenerate;
        app.hf_model = true;
        app.model = "owner/model".to_owned();
        app.prompt = "   ".to_owned();
        assert!(app
            .build_args()
            .expect_err("empty prompt must fail")
            .contains("prompt"));
        assert!(parse_probability("--top-p", "1.01").is_err());
        assert_eq!(parse_probability("--top-p", "1").unwrap(), 1.0);
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 820.0])
            .with_min_inner_size([760.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Hierarchos Native",
        options,
        Box::new(|_cc| Ok(Box::<NativeApp>::default())),
    )
}
