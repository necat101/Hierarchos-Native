//! Source-first construction for ordinary dense decoders and Mixtral.
//! The payload type makes it impossible for these loaders to accidentally
//! upload a resident matrix before converting it to a package-backed linear.
use super::*;

pub(super) trait HostLinearPayload: Clone {
    fn len(&self) -> usize;
    fn upload(
        self,
        device: &VulkanDevice,
        rows: usize,
        input: usize,
        output: usize,
        bias: Option<Vec<f32>>,
    ) -> Result<VulkanLinear>;
}

impl HostLinearPayload for Vec<f32> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn upload(
        self,
        device: &VulkanDevice,
        _rows: usize,
        input: usize,
        output: usize,
        bias: Option<Vec<f32>>,
    ) -> Result<VulkanLinear> {
        match bias {
            Some(bias) => VulkanLinear::new(device, input, output, &self, &bias),
            None => VulkanLinear::new_no_bias(device, input, output, &self),
        }
    }
}

pub(super) fn upload_host_linear<W: HostLinearPayload>(
    device: &VulkanDevice,
    rows: usize,
    input: usize,
    output: usize,
    host: HostLinear<W>,
) -> Result<VulkanLinear> {
    host.weight.upload(device, rows, input, output, host.bias)
}

#[derive(Clone)]
struct PackageLinear {
    source: HostFloatTensorSource,
    offset: usize,
    elements: usize,
    package: Arc<Mutex<ModelPackage>>,
}

impl PackageLinear {
    fn view(
        source: HostFloatTensorSource,
        offset: usize,
        elements: usize,
        package: &Arc<Mutex<ModelPackage>>,
    ) -> Result<Self> {
        let total = source
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .context("streamed matrix shape overflow")?;
        if elements == 0 || offset.checked_add(elements).is_none_or(|end| end > total) {
            bail!("streamed matrix view exceeds its physical tensor");
        }
        if !matches!(source.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
            bail!("streamed matrix source must be F32/F16/BF16");
        }
        Ok(Self {
            source,
            offset,
            elements,
            package: Arc::clone(package),
        })
    }
}

impl HostLinearPayload for PackageLinear {
    fn len(&self) -> usize {
        self.elements
    }
    fn upload(
        self,
        device: &VulkanDevice,
        rows: usize,
        input: usize,
        output: usize,
        bias: Option<Vec<f32>>,
    ) -> Result<VulkanLinear> {
        if input.checked_mul(output) != Some(self.elements) {
            bail!("streamed matrix logical geometry mismatch");
        }
        let package = self.package;
        {
            let mut reader = package
                .lock()
                .map_err(|_| anyhow!("weight package lock poisoned"))?;
            let entry = reader.entry(&self.source.tensor_name)?;
            if entry.shard != self.source.path.canonicalize()?
                || entry.shape != self.source.shape
                || entry.dtype != TensorDtype::from_safetensors(self.source.dtype)?
            {
                bail!("streamed matrix descriptor disagrees with physical metadata");
            }
            // Validate the logical view, never decode a whole packed expert table.
            let tile = crate::weight_offload::DEFAULT_TILE_BYTES as usize / 4;
            for start in (0..self.elements).step_by(tile) {
                let count = tile.min(self.elements - start);
                let values = reader.read_f32_elements(
                    &self.source.tensor_name,
                    self.offset + start,
                    count,
                )?;
                if values.iter().any(|v| !v.is_finite()) {
                    bail!(
                        "streamed matrix {} contains non-finite values",
                        self.source.tensor_name
                    );
                }
            }
        }
        let mut source = TieredLmHead::allocate(
            device,
            self.source.tensor_name,
            input,
            output,
            rows,
            package,
            bias.clone(),
        )?;
        // A contiguous logical matrix may be one slice of a rank-3 expert table.
        source.flat_offset = Some(self.offset);
        VulkanLinear::from_tiered_source(device, Arc::new(source), bias.as_deref())
    }
}

pub(super) fn supports(directory: &Path, config: &VulkanTransformerConfig) -> Result<bool> {
    // Wrappers and special aliases retain their existing architecture-specific
    // loaders until their distinct source geometries have their own gates.
    Ok(matches!(
        (
            config.architecture,
            transformer_package_model_type(directory)?.as_deref()
        ),
        (VulkanTransformerArchitecture::Llama, Some("llama"))
            | (VulkanTransformerArchitecture::Mistral, Some("mistral"))
            | (VulkanTransformerArchitecture::Qwen2, Some("qwen2"))
            | (VulkanTransformerArchitecture::Mixtral, Some("mixtral"))
    ))
}

fn linear(
    store: &mut Gpt2TensorStore,
    package: &Arc<Mutex<ModelPackage>>,
    name: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<HostLinear<PackageLinear>> {
    let source = store.take_expected_source(&format!("{name}.weight"), &[output, input])?;
    let elements = input
        .checked_mul(output)
        .context("streamed linear size overflow")?;
    Ok(HostLinear {
        weight: PackageLinear::view(source, 0, elements, package)?,
        bias: if bias {
            Some(store.take_expected(&format!("{name}.bias"), &[output])?)
        } else {
            None
        },
    })
}

fn norm(store: &mut Gpt2TensorStore, name: &str, hidden: usize) -> Result<HostLayerNorm> {
    Ok(HostLayerNorm {
        weight: store.take_expected(&format!("{name}.weight"), &[hidden])?,
        bias: Vec::new(),
    })
}

fn experts(
    store: &mut Gpt2TensorStore,
    package: &Arc<Mutex<ModelPackage>>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    count: usize,
) -> Result<Vec<HostMoeExpert<PackageLinear>>> {
    let packed = format!("{prefix}.mlp.experts.gate_up_proj");
    let mut result = Vec::with_capacity(count);
    if store.contains(&packed) {
        let twice = intermediate
            .checked_mul(2)
            .context("expert width overflow")?;
        let gate_up = store.take_expected_source(&packed, &[count, twice, hidden])?;
        let down = store.take_expected_source(
            &format!("{prefix}.mlp.experts.down_proj"),
            &[count, hidden, intermediate],
        )?;
        let size = hidden
            .checked_mul(intermediate)
            .context("expert size overflow")?;
        for e in 0..count {
            let projection = |source: &HostFloatTensorSource, offset| -> Result<_> {
                Ok(HostLinear {
                    weight: PackageLinear::view(source.clone(), offset, size, package)?,
                    bias: None,
                })
            };
            result.push(HostMoeExpert {
                gate: Some(projection(&gate_up, e * 2 * size)?),
                up: projection(&gate_up, (e * 2 + 1) * size)?,
                down: projection(&down, e * size)?,
            });
        }
    } else {
        for e in 0..count {
            let current = format!("{prefix}.mlp.experts.{e}");
            let (p, gate, up, down) = if store.contains(&format!("{current}.gate_proj.weight")) {
                (current, "gate_proj", "up_proj", "down_proj")
            } else {
                (
                    format!("{prefix}.block_sparse_moe.experts.{e}"),
                    "w1",
                    "w3",
                    "w2",
                )
            };
            result.push(HostMoeExpert {
                gate: Some(linear(
                    store,
                    package,
                    &format!("{p}.{gate}"),
                    hidden,
                    intermediate,
                    false,
                )?),
                up: linear(
                    store,
                    package,
                    &format!("{p}.{up}"),
                    hidden,
                    intermediate,
                    false,
                )?,
                down: linear(
                    store,
                    package,
                    &format!("{p}.{down}"),
                    intermediate,
                    hidden,
                    false,
                )?,
            });
        }
    }
    Ok(result)
}

fn load(
    directory: &Path,
    config: &VulkanTransformerConfig,
) -> Result<(
    HostTransformerWeights<PackageLinear>,
    Option<Vec<HostMoeLayer<PackageLinear>>>,
)> {
    let mut store = load_llama_layout_tensor_store(directory)?;
    // One body cache per model, not per shard. The tensor descriptor still
    // proves that each physical name resolves to the loader's original file.
    let package = shared_package(directory)?;
    let prefix = [
        "model",
        "model.language_model",
        "model.text_model",
        "language_model",
        "",
    ]
    .into_iter()
    .find(|p| {
        store.contains(&if p.is_empty() {
            "embed_tokens.weight".into()
        } else {
            format!("{p}.embed_tokens.weight")
        })
    })
    .context("missing text embedding source")?;
    let name = |suffix: &str| {
        if prefix.is_empty() {
            suffix.to_owned()
        } else {
            format!("{prefix}.{suffix}")
        }
    };
    let d = config.hidden_size;
    let raw = transformer_text_config_from_package(directory)?;
    let configurable_bias = config.architecture.uses_configurable_attention_bias()
        && raw
            .get("attention_bias")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
    let qkv_bias = configurable_bias
        || (config.architecture == VulkanTransformerArchitecture::Qwen2
            && raw
                .get("attention_bias")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true));
    let mlp_bias = config.architecture.uses_configurable_mlp_bias()
        && raw
            .get("mlp_bias")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
    let embedding =
        store.take_expected_source(&name("embed_tokens.weight"), &[config.vocab_size, d])?;
    let has_lm_head_alias = store.contains("lm_head.weight") && config.tie_word_embeddings;
    let head = if config.tie_word_embeddings {
        if has_lm_head_alias {
            let alias = store.take_expected_source("lm_head.weight", &[config.vocab_size, d])?;
            require_equal_tied_package_sources(&embedding, &alias)?;
        }
        HostPackageLmHead {
            source: embedding,
            tied: true,
            embedding_source: None,
        }
    } else {
        HostPackageLmHead {
            source: store.take_expected_source("lm_head.weight", &[config.vocab_size, d])?,
            tied: false,
            embedding_source: Some(embedding),
        }
    };
    let mut layers = Vec::with_capacity(config.num_layers);
    let mut moe_layers = Vec::new();
    for layer in 0..config.num_layers {
        let p = name(&format!("layers.{layer}"));
        let i = config.layer_intermediate_size(layer);
        let (gate, up, down) = if config.layer_uses_moe(layer) {
            let moe = config.moe.as_ref().context("missing Mixtral MoE config")?;
            let mut all = experts(&mut store, &package, &p, d, i, moe.num_experts)?.into_iter();
            let first = all.next().context("Mixtral requires at least one expert")?;
            let router = if store.contains(&format!("{p}.mlp.gate.weight")) {
                format!("{p}.mlp.gate")
            } else {
                format!("{p}.block_sparse_moe.gate")
            };
            moe_layers.push(HostMoeLayer {
                router: linear(&mut store, &package, &router, d, moe.num_experts, false)?,
                router_correction_bias: vec![0.0; moe.num_experts],
                hash_router: None,
                expert_capacity: 0,
                owned_expert0: None,
                identity_expert_start: moe.num_experts,
                experts: all.collect(),
                shared_expert: None,
                shared_expert_gate: None,
                latent: None,
            });
            (first.gate, first.up, first.down)
        } else {
            (
                Some(linear(
                    &mut store,
                    &package,
                    &format!("{p}.mlp.gate_proj"),
                    d,
                    i,
                    mlp_bias,
                )?),
                linear(
                    &mut store,
                    &package,
                    &format!("{p}.mlp.up_proj"),
                    d,
                    i,
                    mlp_bias,
                )?,
                linear(
                    &mut store,
                    &package,
                    &format!("{p}.mlp.down_proj"),
                    i,
                    d,
                    mlp_bias,
                )?,
            )
        };
        layers.push(HostTransformerLayer {
            post_attention_norm: None,
            post_mlp_norm: None,
            ln1: norm(&mut store, &format!("{p}.input_layernorm"), d)?,
            c_attn: None,
            q_proj: Some(linear(
                &mut store,
                &package,
                &format!("{p}.self_attn.q_proj"),
                d,
                config.query_hidden_size(),
                qkv_bias,
            )?),
            k_proj: Some(linear(
                &mut store,
                &package,
                &format!("{p}.self_attn.k_proj"),
                d,
                config.key_value_hidden_size(),
                qkv_bias,
            )?),
            v_proj: Some(linear(
                &mut store,
                &package,
                &format!("{p}.self_attn.v_proj"),
                d,
                config.key_value_hidden_size(),
                qkv_bias,
            )?),
            q_norm: None,
            k_norm: None,
            c_proj: linear(
                &mut store,
                &package,
                &format!("{p}.self_attn.o_proj"),
                config.query_hidden_size(),
                d,
                configurable_bias,
            )?,
            ln2: norm(&mut store, &format!("{p}.post_attention_layernorm"), d)?,
            c_fc: up,
            c_gate: gate,
            c_mlp_proj: down,
            xielu: None,
        });
    }
    Ok((
        HostTransformerWeights {
            package_lm_head: Some(head),
            token_embedding: Vec::new(),
            position_embedding: None,
            embedding_norm: None,
            layers,
            final_norm: norm(&mut store, &name("norm"), d)?,
            prediction_head: None,
            has_lm_head_alias,
            lm_head: None,
            lm_head_bias: None,
        },
        if moe_layers.is_empty() {
            None
        } else {
            Some(moe_layers)
        },
    ))
}

pub(super) fn build(
    device: VulkanDevice,
    directory: &Path,
    config: VulkanTransformerConfig,
    batch_size: usize,
    seq_len: usize,
) -> Result<VulkanTransformer> {
    config.validate()?;
    if !supports(directory, &config)? {
        bail!("unsupported source-first architecture");
    }
    let rotary_layers = transformer_rotary_layers_for_package(directory, &config)?;
    let padding = transformer_token_embedding_padding_idx_for_package(directory, &config)?;
    let positions_trainable =
        transformer_position_embeddings_trainable_for_package(directory, &config)?;
    let layerdrop = transformer_layerdrop_for_package(directory, &config)?;
    let source_model_type = transformer_package_model_type(directory)?;
    let causal_attention = config.uses_causal_attention();
    let (host, moe) = load(directory, &config)?;
    let mut graph = VulkanTransformer::new(
        device,              // device
        config,              // config
        None,                // albert_topology
        batch_size,          // batch_size
        seq_len,             // seq_len
        seq_len,             // encoder_seq_len
        causal_attention,    // causal_attention
        source_model_type,   // source_model_type
        host,                // host
        rotary_layers,       // rotary_layers
        None,                // attention_sinks
        None,                // attention_value_scale
        None,                // host_attention_gates
        None,                // host_qwen_gated_delta_nets
        None,                // host_deepseek_v4_attentions
        None,                // host_deepseek_v4_hyper_connections
        padding,             // token_embedding_padding_idx
        None,                // esm_mask_token_id
        positions_trainable, // position_embeddings_trainable
        layerdrop,           // layerdrop
        None,                // token_type_embedding
        None,                // auxiliary_embedding
        None,                // roc_bert_embedding_frontend
        None,                // embedding_project
        None,                // output_project
        moe,                 // host_moe_layers
        None,                // host_mla_attentions
        None,                // host_dsa_indexers
        None,                // host_minimax_m3_indexers
        None,                // host_qwen4_qsa_indexers
        None,                // host_qwen4_hyper_connections
        None,                // cross_attentions
        None,                // relative_attention_bias
    )?;
    let mut count = 0;
    graph.for_each_frozen_graph_linear_mut(|_, _, linear| {
        if linear.weight.tiered_source().is_none() {
            bail!("source-first graph contains a resident projection");
        }
        count += 1;
        Ok(())
    })?;
    if count == 0 {
        bail!("source-first graph contains no projections");
    }
    graph.streamed_linears = count;
    graph.source_first_parameters = true;
    Ok(graph)
}

pub(super) fn validation_rows(columns: usize) -> Result<usize> {
    let bytes = columns
        .checked_mul(4)
        .filter(|n| *n != 0)
        .context("invalid validation row width")?;
    Ok((crate::weight_offload::DEFAULT_TILE_BYTES as usize / bytes)
        .max(1)
        .min(TIERED_LM_HEAD_TILE_ROWS))
}
