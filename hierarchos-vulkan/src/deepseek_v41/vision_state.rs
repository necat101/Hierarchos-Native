//! Atomic joint checkpoints and native image + standard PEFT inference exports.
use super::*;
const IMAGE_FILE: &str = "native_image.safetensors";
const METADATA_FILE: &str = "multimodal_state.json";
const MATH: &str = "windows_aten_intel_avx2_v1";
#[derive(Serialize, Deserialize)]
struct JointState {
    schema: u32,
    kind: String,
    math: String,
    base_sha256: String,
    adapter_snapshot_sha256: String,
    image_file_sha256: String,
    step: u64,
    hyper: DeepseekV41AdamWConfig,
    options: DeepseekV41ModelTrainingConfig,
    #[serde(default = "reference_precision")]
    wmma_mode: String,
}
fn reference_precision() -> String { "off".into() }
fn image_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
impl DeepseekV41MultimodalTrainer {
    /// Save both optimizers, all native image weights and standard PEFT factors.
    /// The frozen base is referenced by its complete existing identity binding.
    pub fn save_checkpoint(&self, directory: impl AsRef<Path>) -> Result<()> {
        self.publish_native(directory.as_ref(), true)
    }
    /// Export trained native vision/delimiter weights alongside standard PEFT.
    /// This overlay references the unchanged base; no optimizer state is included.
    pub fn export_native(&self, directory: impl AsRef<Path>) -> Result<()> {
        self.publish_native(directory.as_ref(), false)
    }
    fn publish_native(&self, directory: &Path, optimizer: bool) -> Result<()> {
        self.language.unchanged()?;
        if self.optimizer_step() != self.image.step() as u64 {
            bail!("joint optimizer counters disagree");
        }
        if directory.exists() {
            bail!("native multimodal destination already exists");
        }
        let parent = directory
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let parent = parent.canonicalize()?;
        let destination = parent.join(
            directory
                .file_name()
                .context("native multimodal destination has no name")?,
        );
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let staging = parent.join(format!(".v41-native-{}-{nonce}", std::process::id()));
        fs::create_dir(&staging)?;
        let owned = staging.canonicalize()?;
        let result = (|| -> Result<()> {
            let language = staging.join("language");
            if optimizer {
                self.language.save_checkpoint(&language)?;
            } else {
                self.language.export_adapter(&language)?;
            }
            self.image
                .save_native_state(&staging.join(IMAGE_FILE), optimizer)?;
            let metadata = JointState {
                schema: 1,
                kind: if optimizer { "training" } else { "export" }.to_owned(),
                math: MATH.to_owned(),
                base_sha256: self.language.identity.clone(),
                adapter_snapshot_sha256: self.adapter()?.snapshot_digest()?,
                image_file_sha256: image_hash(&staging.join(IMAGE_FILE))?,
                step: if optimizer { self.optimizer_step() } else { 0 },
                hyper: self.hyper,
                options: self.language.options.clone(),
                wmma_mode: self.image.precision_mode().into(),
            };
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(staging.join(METADATA_FILE))?;
            file.write_all(&serde_json::to_vec_pretty(&metadata)?)?;
            file.sync_all()?;
            drop(file);
            // Validate PEFT and optimizer serialization on the host. The image
            // writer checks every serialized tensor without a second GPU graph.
            let adapter = if optimizer {
                DeepseekV41AdapterTrainer::resume_from_checkpoint(&language)?.adapter()?
            } else {
                DeepseekV41LoraAdapter::load(&language)?
            };
            if adapter.snapshot_digest()? != metadata.adapter_snapshot_sha256 {
                bail!("native export adapter verification failed");
            }
            fs::rename(&staging, &destination).context("publishing native multimodal state")?;
            Ok(())
        })();
        if result.is_err()
            && owned.parent() == Some(parent.as_path())
            && staging.canonicalize().ok().as_ref() == Some(&owned)
        {
            let _ = fs::remove_dir_all(&owned);
        }
        result
    }
    pub fn resume(
        device: VulkanDevice,
        base: impl AsRef<Path>,
        checkpoint: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::load_native(device, base.as_ref(), checkpoint.as_ref(), true)
    }
    pub fn from_native_export(
        device: VulkanDevice,
        base: impl AsRef<Path>,
        export: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::load_native(device, base.as_ref(), export.as_ref(), false)
    }
    fn load_native(
        device: VulkanDevice,
        base: &Path,
        directory: &Path,
        optimizer: bool,
    ) -> Result<Self> {
        let _math = crate::deepseek_v41::cpu_math::avx2();
        let metadata: JointState =
            serde_json::from_slice(&fs::read(directory.join(METADATA_FILE))?)?;
        if optimizer && metadata.wmma_mode != crate::wmma::WmmaMode::from_env()?.label() {
            bail!("native multimodal checkpoint precision mode mismatch");
        }
        if metadata.schema != 1
            || metadata.kind != (if optimizer { "training" } else { "export" })
            || metadata.math != MATH
            || image_hash(&directory.join(IMAGE_FILE))? != metadata.image_file_sha256
        {
            bail!("native multimodal state kind, policy or integrity mismatch");
        }
        let expected_options = serde_json::to_value(&metadata.options)?;
        let language = if optimizer {
            DeepseekV41ModelTrainer::resume(device.clone(), base, directory.join("language"))?
        } else {
            let adapter = DeepseekV41LoraAdapter::load(directory.join("language"))?;
            DeepseekV41ModelTrainer::new(
                device.clone(),
                base,
                &adapter,
                metadata.hyper,
                metadata.options,
            )?
        };
        if serde_json::to_value(&language.options)? != expected_options
            || language.identity != metadata.base_sha256
            || language.optimizer.step_count() != metadata.step
            || language.optimizer.config() != metadata.hyper
            || language.adapter()?.snapshot_digest()? != metadata.adapter_snapshot_sha256
        {
            bail!("native multimodal state belongs to different frozen weights, factors or optimizer settings");
        }
        let c = crate::DeepseekV41VisionConfig::from_source_config(
            &language.package.config.source_config,
        )?;
        let image = VulkanDeepseekV41Image::load_native_state(
            device,
            &directory.join(IMAGE_FILE),
            &c,
            optimizer,
            u32::try_from(metadata.step).context("native image step overflow")?,
        )?;
        let hc = language.package.config.text.hc_mult;
        let image_token_id = language.package.config.source_config["image_token_id"]
            .as_u64()
            .context("native image token id missing")?;
        if image_token_id >= language.package.config.text.vocab_size as u64 {
            bail!("native image token id exceeds vocabulary");
        }
        Ok(Self {
            language,
            image,
            hyper: metadata.hyper,
            image_token_id: image_token_id as u32,
            hc,
            pending: None,
        })
    }
}
