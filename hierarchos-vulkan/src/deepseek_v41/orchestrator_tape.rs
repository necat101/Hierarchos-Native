//! Retained layer execution with the actual indexer and source publications.
use super::*;
use crate::deepseek_v41::{
    deepseek_v41_window_indices, DeepseekV41AdapterGradient, DeepseekV41AttentionCacheId,
    DeepseekV41AttentionOutputTape, DeepseekV41AttentionReadId, DeepseekV41AttentionReverseTape,
    DeepseekV41AttentionTape, DeepseekV41AttentionWindowView, DeepseekV41AttentionWriteId,
    DeepseekV41CompressorAdjoints, DeepseekV41SparseAttentionParams,
};

pub(crate) struct AttentionRecording {
    pub cache: DeepseekV41AttentionTape,
    pub windows: Vec<DeepseekV41AttentionCacheId>,
    pub compressed: Vec<Option<DeepseekV41AttentionCacheId>>,
    batch: usize,
    hidden: usize,
    window: usize,
    device: VulkanDevice,
    capacities: Vec<usize>,
}
pub(crate) struct AttentionLayerTape {
    input: DeepseekV41AttentionInputTape,
    output: DeepseekV41AttentionOutputTape,
    read: DeepseekV41AttentionReadId,
    window_write: DeepseekV41AttentionWriteId,
    compressor: Option<DeepseekV41CompressorTape>,
    finalizer: Option<DeepseekV41CompressedKvTape>,
    compressed_write: Option<DeepseekV41AttentionWriteId>,
    layer: usize,
    input_len: usize,
}

pub(super) fn finish_forward(
    rec: &mut AttentionRecording,
    adjoints: &DeepseekV41AttentionProjectionAdjoints,
    attention: &mut DeepseekV41Attention,
    input: DeepseekV41AttentionInputTape,
    plan: &DeepseekV41LayerPlan,
    step: Step,
    selection: Option<&DeepseekV41IndexSelection>,
    compressor: Option<DeepseekV41CompressorTape>,
    finalizer: Option<DeepseekV41CompressedKvTape>,
    compressed_write: Option<DeepseekV41AttentionWriteId>,
) -> Result<(GpuBuffer, Option<AttentionLayerTape>)> {
    let layer = plan.layer_id;
    let (window_write, view) = if step.prefill {
        let write = rec
            .cache
            .seed_window(&rec.windows[layer], &input.window_kv, step.seq)?;
        let view = DeepseekV41AttentionWindowView::Prefill(write.clone());
        (write, view)
    } else {
        let write =
            rec.cache
                .write_window(&rec.windows[layer], &input.window_kv, step.start_pos)?;
        (
            write,
            DeepseekV41AttentionWindowView::Cache(rec.windows[layer].clone()),
        )
    };
    let (wi, window_topk) =
        deepseek_v41_window_indices(rec.batch, step.seq, step.start_pos, rec.window)?;
    let wi = GpuBuffer::from_u32_scheduled(
        &rec.cache_device(),
        &wi.into_iter().map(|n| n as u32).collect::<Vec<_>>(),
    )?;
    let empty = GpuBuffer::zeros_u32(&rec.cache_device(), 1)?;
    let ci = selection.and_then(|s| s.indices.as_ref()).unwrap_or(&empty);
    let compressed = plan
        .kv_source_layer
        .and_then(|source| rec.compressed[source].as_ref());
    let capacity = compressed.map_or(0, |_| rec.capacity(plan.kv_source_layer.unwrap()));
    let read = rec.cache.attend(
        attention,
        &input.query,
        view,
        compressed,
        &wi,
        ci,
        DeepseekV41SparseAttentionParams {
            batch: rec.batch,
            seq: step.seq,
            window_rows: if step.prefill { step.seq } else { rec.window },
            window_topk,
            compressed_topk: selection.map_or(0, |s| s.topk),
            compressed_capacity: capacity,
        },
    )?;
    let output =
        adjoints.forward_output(attention, &read.output, rec.batch, step.seq, step.start_pos)?;
    attention.mark_retained_step(rec.batch, step.seq, step.start_pos);
    Ok((
        output.output.clone(),
        Some(AttentionLayerTape {
            input,
            output,
            read: read.id,
            window_write,
            compressor,
            finalizer,
            compressed_write,
            layer,
            input_len: rec.batch * step.seq * rec.hidden,
        }),
    ))
}

impl AttentionRecording {
    fn cache_device(&self) -> VulkanDevice {
        self.device.clone()
    }
    fn capacity(&self, source: usize) -> usize {
        self.capacities[source]
    }
}

impl DeepseekV41AttentionOrchestrator {
    pub(crate) fn recording(&self) -> Result<AttentionRecording> {
        let mut cache = DeepseekV41AttentionTape::new(self.device.clone());
        let windows = self
            .layers
            .iter()
            .map(|l| cache.register_window(l.attention.window_cache()))
            .collect::<Result<Vec<_>>>()?;
        let compressed = self
            .compressed
            .iter()
            .map(|c| c.as_ref().map(|c| cache.register_compressed(c)).transpose())
            .collect::<Result<Vec<_>>>()?;
        Ok(AttentionRecording {
            cache,
            windows,
            compressed,
            batch: self.runtime.batch_size,
            hidden: self.hidden,
            window: self.window,
            device: self.device.clone(),
            capacities: self
                .compressed
                .iter()
                .map(|c| c.as_ref().map_or(0, |c| c.capacity()))
                .collect(),
        })
    }

    pub(crate) fn backward_layer(
        &mut self,
        tape: &AttentionLayerTape,
        reverse: &mut DeepseekV41AttentionReverseTape,
        projection: &DeepseekV41AttentionProjectionAdjoints,
        compressor: &DeepseekV41CompressorAdjoints,
        state: &mut Option<(GpuBuffer, GpuBuffer)>,
        upstream: &GpuBuffer,
    ) -> Result<(GpuBuffer, Vec<DeepseekV41AdapterGradient>)> {
        let layer = &mut self.layers[tape.layer];
        let result = (|| {
            layer.attention.load_projections()?;
            if tape.compressor.is_some() {
                layer
                    .compressor
                    .as_mut()
                    .context("V4.1 retained source lost its compressor")?
                    .load_weights()?;
            }
            let output_grad =
                projection.backward_output(&layer.attention, &tape.output, upstream)?;
            let mut adapters =
                output_grad.adapter_gradients(&format!("layers.{}.attn", tape.layer))?;
            reverse.backward_read(&tape.read, output_grad.grad_input.as_ref())?;
            let q = reverse.take_query(&tape.read)?;
            let kv = reverse.take_source(&tape.window_write)?;
            let input_grad = projection.backward_inputs(
                &layer.attention,
                &tape.input,
                q.as_ref(),
                kv.as_ref(),
                None,
            )?;
            adapters.extend(input_grad.adapter_gradients(&format!("layers.{}.attn", tape.layer))?);
            let mut input = input_grad
                .grad_input
                .context("V4.1 retained attention has no input gradient")?;
            if let Some(tape_c) = &tape.compressor {
                let finalized = tape
                    .compressed_write
                    .as_ref()
                    .map(|id| reverse.take_source(id))
                    .transpose()?
                    .flatten();
                let latent = finalized
                    .as_ref()
                    .map(|g| {
                        layer
                            .finalizer
                            .as_ref()
                            .context("V4.1 retained source lost its finalizer")?
                            .backward(
                                tape.finalizer.as_ref().context(
                                    "V4.1 retained compressed write has no finalizer tape",
                                )?,
                                g,
                            )
                    })
                    .transpose()?;
                if latent.is_some() || state.is_some() {
                    let grad = tape_c.backward(
                        layer.compressor.as_ref().unwrap(),
                        compressor,
                        latent.as_ref(),
                        state.as_ref().map(|(k, s)| (k, s)),
                    )?;
                    input = projection.add(&input, &grad.grad_input, tape.input_len)?;
                    *state = match (grad.grad_state_kv, grad.grad_state_score) {
                        (Some(k), Some(s)) => Some((k, s)),
                        _ => None,
                    };
                }
            }
            Ok((input, adapters))
        })();
        layer.release_projections();
        result
    }
}
