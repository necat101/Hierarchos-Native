//! The complete shifted-HC/Engram/attention/MoE reverse graph.
use super::*;
use crate::deepseek_v41::orchestrator::{AttentionLayerTape, AttentionRecording};
use crate::deepseek_v41::{
    DeepseekV41AdapterGradient, DeepseekV41AttentionProjectionAdjoints,
    DeepseekV41AttentionReverseTape, DeepseekV41CompressorAdjoints, DeepseekV41EngramAdjoints,
    DeepseekV41EngramTape,
};

pub(crate) struct ModelAdjoints {
    pub block: DeepseekV41BlockAdjoints,
    pub attention: DeepseekV41AttentionProjectionAdjoints,
    pub compressor: DeepseekV41CompressorAdjoints,
    pub engram: DeepseekV41EngramAdjoints,
    pub moe: DeepseekV41MixedMoeAdjoints,
}
impl ModelAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            block: DeepseekV41BlockAdjoints::new(device.clone())?,
            attention: DeepseekV41AttentionProjectionAdjoints::new(device.clone())?,
            compressor: DeepseekV41CompressorAdjoints::new(device.clone())?,
            engram: DeepseekV41EngramAdjoints::new(device.clone())?,
            moe: DeepseekV41MixedMoeAdjoints::new(device)?,
        })
    }
    fn add(&self, a: &GpuBuffer, b: &GpuBuffer, n: usize) -> Result<GpuBuffer> {
        self.attention.add(a, b, n)
    }
}
enum MoeTape {
    Fp32(GpuBuffer),
    // MoE has no mutable state. Keep one normalized activation and recompute
    // selected expert operands when this layer is reached in reverse. Frozen
    // expert payloads and low-rank snapshots are never pinned across layers.
    Mixed(GpuBuffer),
}
struct LayerTape {
    layer: usize,
    stream: GpuBuffer,
    previous_pre: GpuBuffer,
    engram: Option<DeepseekV41EngramTape>,
    attn: DeepseekV41HyperConnectionMixes,
    attn_collapse: GpuBuffer,
    attended: GpuBuffer,
    attention: AttentionLayerTape,
    after_attention: GpuBuffer,
    ffn: DeepseekV41HyperConnectionMixes,
    ffn_collapse: GpuBuffer,
    moe: MoeTape,
    mixed: GpuBuffer,
}
pub(crate) struct BackboneStepTape {
    #[cfg(test)]
    position: usize,
    layers: Vec<LayerTape>,
    rows: usize,
    final_stream: GpuBuffer,
    final_pre: GpuBuffer,
}
impl DeepseekV41Backbone {
    pub(crate) fn recording(&self) -> Result<AttentionRecording> {
        self.attention.recording()
    }

    pub(crate) fn forward_retained(
        &mut self,
        stream: &GpuBuffer,
        seq: usize,
        position: usize,
        token_ids: Option<&GpuBuffer>,
        mask: Option<&GpuBuffer>,
        kernels: &ModelAdjoints,
        recording: &mut AttentionRecording,
    ) -> Result<(GpuBuffer, BackboneStepTape)> {
        if position == 0 {
            self.attention.begin_prefill(seq)?;
        } else {
            self.attention.begin_decode(position)?;
        }
        self.begin_engram(token_ids, mask, position, seq)?;
        let rows = self.rows()?;
        let width = checked_size("retained backbone stream", &[rows, self.hc, self.hidden])?;
        if stream.f32_capacity() < width || !stream.belongs_to_device(&self.device) {
            bail!("V4.1 retained backbone input geometry/device mismatch");
        }
        let mut current = stream.clone();
        let mut previous_pre = self.hyper[0].identity_pre_mix(rows)?;
        let mut layers = Vec::with_capacity(self.hyper.len());
        for layer in 0..self.hyper.len() {
            let stream = self.cast.cast_buffer(&current, width, self.precision)?;
            let engram = self
                .engram
                .as_ref()
                .map(|e| e.forward_retained(layer, &stream, rows, &kernels.engram))
                .transpose()?
                .flatten();
            let stream = self.cast.cast_buffer(
                engram.as_ref().map_or(&stream, |e| &e.output),
                width,
                self.precision,
            )?;
            let attn = self.hyper[layer].attention_mixes(&stream, rows)?;
            let collapsed = self.hyper[layer].pre(&stream, &previous_pre, rows)?;
            let attn_collapse =
                self.cast
                    .cast_buffer(&collapsed, rows * self.hidden, self.precision)?;
            let normalized = self.norm.apply(
                &self.device,
                &attn_collapse,
                &self.attn_norm[layer],
                rows,
                self.hidden,
                self.norm_eps,
            )?;
            let normalized =
                self.cast
                    .cast_buffer(&normalized, rows * self.hidden, self.precision)?;
            let (attended, attention) = self.attention.forward_layer_recorded(
                layer,
                &normalized,
                Some((&kernels.attention, recording)),
            )?;
            #[cfg(test)]
            {
                // The two-level index selection is the graph's only other
                // discrete edge besides routing; record its scores, candidate
                // mask and ranking so a flip can be localized to a stage.
                if let Ok(Some(selection)) = self.attention.selection(layer) {
                    if selection.width > 0 {
                        crate::deepseek_v41::training_trace::record(
                            &format!("{position}/{layer}/index_scores"),
                            &selection.scores,
                            rows * selection.width,
                        );
                        if let Some(indices) = selection.indices.as_ref() {
                            crate::deepseek_v41::training_trace::record(
                                &format!("{position}/{layer}/index_topk"),
                                indices,
                                rows * selection.topk,
                            );
                        }
                        if let Some(candidates) = selection.candidates.as_ref() {
                            crate::deepseek_v41::training_trace::record(
                                &format!("{position}/{layer}/index_candidates"),
                                candidates,
                                rows * selection.width,
                            );
                        }
                    }
                }
            }
            let after = self.hyper[layer].post(&attended, &stream, &attn.post, &attn.comb, rows)?;
            let after_attention = self.cast.cast_buffer(&after, width, self.precision)?;
            let ffn = self.hyper[layer].ffn_mixes(&after_attention, rows)?;
            let collapsed = self.hyper[layer].pre(&after_attention, &attn.pre, rows)?;
            let ffn_collapse =
                self.cast
                    .cast_buffer(&collapsed, rows * self.hidden, self.precision)?;
            let normalized = self.norm.apply(
                &self.device,
                &ffn_collapse,
                &self.ffn_norm[layer],
                rows,
                self.hidden,
                self.norm_eps,
            )?;
            let normalized =
                self.cast
                    .cast_buffer(&normalized, rows * self.hidden, self.precision)?;
            let (mixed, moe) = match &self.moe[layer] {
                MoeExecution::Fp32(moe) => {
                    (moe.apply(&normalized, rows)?, MoeTape::Fp32(normalized))
                }
                MoeExecution::Mixed(moe) => {
                    (moe.apply(&normalized, rows)?, MoeTape::Mixed(normalized))
                }
            };
            let output =
                self.hyper[layer].post(&mixed, &after_attention, &ffn.post, &ffn.comb, rows)?;
            current = self.cast.cast_buffer(&output, width, self.precision)?;
            #[cfg(test)]
            {
                crate::deepseek_v41::training_trace::record(
                    &format!("{position}/{layer}/attended"),
                    &attended,
                    rows * self.hidden,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{position}/{layer}/after_attention"),
                    &after_attention,
                    width,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{position}/{layer}/mixed"),
                    &mixed,
                    rows * self.hidden,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{position}/{layer}/output"),
                    &current,
                    width,
                );
            }
            let next_pre = ffn.pre.clone();
            layers.push(LayerTape {
                layer,
                stream,
                previous_pre,
                engram,
                attn,
                attn_collapse,
                attended,
                attention: attention.unwrap(),
                after_attention,
                ffn,
                ffn_collapse,
                moe,
                mixed,
            });
            previous_pre = next_pre;
        }
        let collapsed = self.hyper[0].pre(&current, &previous_pre, rows)?;
        let output = self
            .cast
            .cast_buffer(&collapsed, rows * self.hidden, self.precision)?;
        Ok((
            output,
            BackboneStepTape {
                #[cfg(test)]
                position,
                layers,
                rows,
                final_stream: current,
                final_pre: previous_pre,
            },
        ))
    }

    pub(crate) fn backward_retained(
        &mut self,
        tape: BackboneStepTape,
        upstream: &GpuBuffer,
        reverse: &mut DeepseekV41AttentionReverseTape,
        states: &mut [Option<(GpuBuffer, GpuBuffer)>],
        kernels: &ModelAdjoints,
    ) -> Result<(GpuBuffer, Vec<DeepseekV41AdapterGradient>)> {
        let rows = tape.rows;
        let n = rows * self.hc * self.hidden;
        let end = kernels.block.hc_pre(
            upstream,
            &tape.final_stream,
            &tape.final_pre,
            rows,
            self.hc,
            self.hidden,
        )?;
        let mut stream_grad = end.grad_stream;
        let mut pre_grad = end.grad_pre;
        let mut adapters = Vec::new();
        for t in tape.layers.into_iter().rev() {
            let l = t.layer;
            #[cfg(test)]
            crate::deepseek_v41::training_trace::record(
                &format!("{}/{l}/output/grad", tape.position),
                &stream_grad,
                n,
            );
            let post = kernels.block.hc_post(
                &self.hyper[l],
                &stream_grad,
                &t.mixed,
                &t.after_attention,
                &t.ffn.post,
                &t.ffn.comb,
                rows,
            )?;
            #[cfg(test)]
            crate::deepseek_v41::training_trace::record(
                &format!("{}/{l}/mixed/grad", tape.position),
                &post.grad_x,
                rows * self.hidden,
            );
            let moe = match (&self.moe[l], t.moe) {
                (MoeExecution::Fp32(moe), MoeTape::Fp32(input)) => {
                    kernels.block.moe(moe, &post.grad_x, &input, rows)?
                }
                (MoeExecution::Mixed(moe), MoeTape::Mixed(input)) => {
                    let tape = kernels.moe.forward(moe, &input, rows)?;
                    let grad = kernels.moe.backward(moe, &tape, &post.grad_x)?;
                    adapters.extend(grad.adapter_gradients(&format!("layers.{l}.ffn"))?);
                    grad.grad_input
                }
                _ => bail!("V4.1 retained MoE storage changed"),
            };
            let norm = kernels.block.ffn_norm(
                &moe,
                &t.ffn_collapse,
                &self.ffn_norm[l],
                rows,
                self.hidden,
                self.norm_eps,
            )?;
            #[cfg(test)]
            {
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_norm_upstream", tape.position),
                    &moe,
                    rows * self.hidden,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_norm_input", tape.position),
                    &t.ffn_collapse,
                    rows * self.hidden,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/norm_collapsed/grad", tape.position),
                    &norm,
                    rows * self.hidden,
                );
            }
            let collapse = kernels.block.hc_pre(
                &norm,
                &t.after_attention,
                &t.attn.pre,
                rows,
                self.hc,
                self.hidden,
            )?;
            let split = kernels.block.hc_split(
                &self.hyper[l],
                DeepseekV41HyperStage::FeedForward,
                &t.ffn.projection,
                &t.after_attention,
                &pre_grad,
                &post.grad_post,
                &post.grad_comb,
                rows,
            )?;
            let mix = kernels.block.hc_mixes(
                &self.hyper[l],
                DeepseekV41HyperStage::FeedForward,
                &split,
                &t.ffn.projection,
                &t.after_attention,
                rows,
            )?;
            let after = kernels.add(&post.grad_residual, &collapse.grad_stream, n)?;
            let after = kernels.add(&after, &mix, n)?;
            #[cfg(test)]
            {
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/after_attention/grad", tape.position),
                    &after,
                    n,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_residual/grad", tape.position),
                    &post.grad_residual,
                    n,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_collapse/grad", tape.position),
                    &collapse.grad_stream,
                    n,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_mix/grad", tape.position),
                    &mix,
                    n,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_pre/grad", tape.position),
                    &pre_grad,
                    rows * self.hc,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_post/grad", tape.position),
                    &post.grad_post,
                    rows * self.hc,
                );
                crate::deepseek_v41::training_trace::record(
                    &format!("{}/{l}/ffn_comb/grad", tape.position),
                    &post.grad_comb,
                    rows * self.hc * self.hc,
                );
            }
            let post = kernels.block.hc_post(
                &self.hyper[l],
                &after,
                &t.attended,
                &t.stream,
                &t.attn.post,
                &t.attn.comb,
                rows,
            )?;
            let (attention, mut attention_adapters) = self.attention.backward_layer(
                &t.attention,
                reverse,
                &kernels.attention,
                &kernels.compressor,
                &mut states[l],
                &post.grad_x,
            )?;
            #[cfg(test)]
            crate::deepseek_v41::training_trace::record(
                &format!("{}/{l}/attended/grad", tape.position),
                &post.grad_x,
                rows * self.hidden,
            );
            adapters.append(&mut attention_adapters);
            let norm = kernels.block.ffn_norm(
                &attention,
                &t.attn_collapse,
                &self.attn_norm[l],
                rows,
                self.hidden,
                self.norm_eps,
            )?;
            let input_collapse = kernels.block.hc_pre(
                &norm,
                &t.stream,
                &t.previous_pre,
                rows,
                self.hc,
                self.hidden,
            )?;
            let split = kernels.block.hc_split(
                &self.hyper[l],
                DeepseekV41HyperStage::Attention,
                &t.attn.projection,
                &t.stream,
                &collapse.grad_pre,
                &post.grad_post,
                &post.grad_comb,
                rows,
            )?;
            let mix = kernels.block.hc_mixes(
                &self.hyper[l],
                DeepseekV41HyperStage::Attention,
                &split,
                &t.attn.projection,
                &t.stream,
                rows,
            )?;
            let stream = kernels.add(&post.grad_residual, &input_collapse.grad_stream, n)?;
            let stream = kernels.add(&stream, &mix, n)?;
            stream_grad = if let Some(engram) = t.engram {
                let (gradient, mut engram_adapters) = self
                    .engram
                    .as_ref()
                    .context("V4.1 retained Engram runtime missing")?
                    .backward_retained(l, &engram, &stream, &kernels.engram)?;
                adapters.append(&mut engram_adapters);
                gradient
            } else {
                stream
            };
            pre_grad = input_collapse.grad_pre;
        }
        Ok((stream_grad, adapters))
    }
}
