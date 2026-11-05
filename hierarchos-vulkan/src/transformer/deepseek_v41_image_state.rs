//! Lossless native image parameters and AdamW slots in one SafeTensors file.
use super::*;
use safetensors::{
    tensor::{serialize_to_file, Dtype, TensorView},
    SafeTensors,
};
use std::{collections::HashMap, fs, path::Path};

#[derive(Serialize, Deserialize)]
struct ImageState {
    schema: u32,
    kind: String,
    config: DeepseekV41VisionConfig,
    height: usize,
    width: usize,
    step: u32,
}
impl VulkanDeepseekV41Image {
    fn state_parameters(&self) -> Vec<(String, &TrainableParameter, Vec<usize>)> {
        let v = &self.vision;
        let mut result = Vec::new();
        let mut linears = vec![
            ("vision.patch_embed.proj".to_owned(), &v.patch),
            ("aligner.w1".to_owned(), &v.aligner1),
            ("aligner.w2".to_owned(), &v.aligner2),
        ];
        for (i, b) in v.blocks.iter().enumerate() {
            let prefix = format!("vision.blocks.{i}");
            linears.extend([
                (format!("{prefix}.attn.wqkv"), &b.qkv),
                (format!("{prefix}.attn.wo"), &b.output),
                (format!("{prefix}.mlp.w1"), &b.gate_up),
                (format!("{prefix}.mlp.w2"), &b.down),
            ]);
        }
        for (name, l) in linears {
            result.push((
                format!("{name}.weight"),
                &l.weight,
                vec![l.output_dim, l.input_dim],
            ));
            if l.has_bias {
                result.push((format!("{name}.bias"), &l.bias, vec![l.output_dim]));
            }
        }
        for (i, b) in v.blocks.iter().enumerate() {
            result.push((
                format!("vision.blocks.{i}.norm1.weight"),
                &b.norm1.parameter.weight,
                vec![v.config.vision_dim],
            ));
            result.push((
                format!("vision.blocks.{i}.norm2.weight"),
                &b.norm2.parameter.weight,
                vec![v.config.vision_dim],
            ));
        }
        result.push((
            "vision.norm.weight".to_owned(),
            &v.norm.parameter.weight,
            vec![v.config.vision_dim],
        ));
        for (name, p) in MARKERS.iter().zip(&self.markers) {
            result.push(((*name).to_owned(), p, vec![v.config.dim]));
        }
        result
    }
    pub(crate) fn save_native_state(&self, path: &Path, optimizer: bool) -> Result<()> {
        if path.exists() {
            bail!("native image state destination already exists");
        }
        let mut data = BTreeMap::new();
        for (name, p, shape) in self.state_parameters() {
            let mut tensors = vec![(name.clone(), p.read()?)];
            if optimizer {
                tensors.push((
                    format!("{name}.exp_avg"),
                    p.exp_avg
                        .as_ref()
                        .context("native image parameter has no first moment")?
                        .read_f32(p.len)?,
                ));
                tensors.push((
                    format!("{name}.exp_avg_sq"),
                    p.exp_avg_sq
                        .as_ref()
                        .context("native image parameter has no second moment")?
                        .read_f32(p.len)?,
                ));
            }
            for (key, values) in tensors {
                if values.iter().any(|x| !x.is_finite()) {
                    bail!("non-finite native image state at {key}");
                }
                data.insert(
                    key,
                    (
                        shape.clone(),
                        values
                            .iter()
                            .flat_map(|x| x.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                );
            }
        }
        let metadata = ImageState {
            schema: 1,
            kind: if optimizer { "training" } else { "export" }.to_owned(),
            config: self.vision.config.clone(),
            height: self.vision.height,
            width: self.vision.width,
            step: if optimizer { self.step() } else { 0 },
        };
        let metadata = HashMap::from([(
            "hierarchos_native_v41_image".to_owned(),
            serde_json::to_string(&metadata)?,
        )]);
        let views = data
            .iter()
            .map(|(name, (shape, bytes))| {
                Ok((
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, Some(metadata), path)?;
        fs::OpenOptions::new().write(true).open(path)?.sync_all()?;
        // Verify serialized bytes on the host without allocating another ViT.
        let saved_bytes = fs::read(path)?;
        let saved = SafeTensors::deserialize(&saved_bytes)?;
        if saved.len() != data.len() {
            bail!("native image state readback tensor count mismatch");
        }
        for (name, (shape, bytes)) in &data {
            let actual = saved.tensor(name)?;
            if actual.dtype() != Dtype::F32 || actual.shape() != shape || actual.data() != bytes {
                bail!("native image state readback mismatch at {name}");
            }
        }
        Ok(())
    }
    pub(crate) fn load_native_state(
        device: VulkanDevice,
        path: &Path,
        expected: &DeepseekV41VisionConfig,
        optimizer: bool,
        step: u32,
    ) -> Result<Self> {
        let bytes = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let (_, header) = SafeTensors::read_metadata(&bytes)?;
        let metadata: ImageState = serde_json::from_str(
            header
                .metadata()
                .as_ref()
                .and_then(|m| m.get("hierarchos_native_v41_image"))
                .context("native image state metadata missing")?,
        )?;
        if metadata.schema != 1
            || metadata.kind != if optimizer { "training" } else { "export" }
            || metadata.step != step
            || serde_json::to_value(&metadata.config)? != serde_json::to_value(expected)?
        {
            bail!("native image state contract, configuration or optimizer step mismatch");
        }
        let mut initial = Tensors::new();
        let mut slots = BTreeMap::new();
        for (name, tensor) in tensors.tensors() {
            if tensor.dtype() != Dtype::F32 {
                bail!("native image state must be FP32");
            }
            let values: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if values.iter().any(|x| !x.is_finite())
                || (name.ends_with(".exp_avg_sq") && values.iter().any(|x| *x < 0.))
            {
                bail!("invalid native image optimizer state at {name}");
            }
            if name.ends_with(".exp_avg") || name.ends_with(".exp_avg_sq") {
                if step == 0 && values.iter().any(|x| *x != 0.) {
                    bail!("native image step zero has nonzero moments");
                }
                slots.insert(name, (tensor.shape().to_vec(), values));
            } else {
                initial.insert(
                    name,
                    DeepseekV41VisionTensor {
                        shape: tensor.shape().to_vec(),
                        values,
                    },
                );
            }
        }
        let mut image = Self::from_tensors(
            device,
            metadata.config,
            metadata.height,
            metadata.width,
            &initial,
        )?;
        let params = image.state_parameters();
        if initial.len() != params.len()
            || slots.len() != if optimizer { 2 * params.len() } else { 0 }
        {
            bail!("unexpected native image state tensors");
        }
        if optimizer {
            for (name, p, shape) in params {
                for (suffix, target) in [
                    ("exp_avg", p.exp_avg.as_ref()),
                    ("exp_avg_sq", p.exp_avg_sq.as_ref()),
                ] {
                    let (actual_shape, values) = slots
                        .get(&format!("{name}.{suffix}"))
                        .context("missing native image moment")?;
                    if *actual_shape != shape {
                        bail!("native image moment shape mismatch");
                    }
                    target.context("missing device moment")?.write_f32(values)?;
                }
            }
        }
        image.vision.step = step;
        Ok(image)
    }
}
