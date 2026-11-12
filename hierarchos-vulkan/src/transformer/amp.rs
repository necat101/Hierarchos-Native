//! Device-side dynamic loss scaling for the qualified single-graph trainers.
//! Queue every optimizer parameter before resolving one global overflow flag.
use super::*;
use crate::training_numerics::{
    VulkanDynamicLossScaleController, VulkanDynamicLossScaleSeed, VulkanGradientNonfiniteDetector,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct TransformerLossScaleState {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u64,
    pub growth_tracker: u64,
    pub skipped_steps: u64,
    #[serde(default = "one_microbatch")]
    pub accumulation_steps: u32,
    #[serde(default)]
    pub pending_microbatches: u32,
}
fn one_microbatch() -> u32 {
    1
}

impl Default for TransformerLossScaleState {
    fn default() -> Self {
        Self {
            scale: 256.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
            skipped_steps: 0,
            accumulation_steps: 1,
            pending_microbatches: 0,
        }
    }
}

impl TransformerLossScaleState {
    fn seed(self) -> Result<VulkanDynamicLossScaleSeed> {
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || !self.growth_factor.is_finite()
            || self.growth_factor <= 1.0
            || !self.backoff_factor.is_finite()
            || !(0.0..1.0).contains(&self.backoff_factor)
            || self.growth_interval == 0
            || !self.scale.recip().is_finite()
            || self.scale.recip() <= 0.0
            || !(self.scale * self.growth_factor).is_finite()
            || self.scale * self.backoff_factor <= 0.0
            || !(1..=1024).contains(&self.accumulation_steps)
            || self.pending_microbatches >= self.accumulation_steps
        {
            bail!("invalid Transformer loss scaler state");
        }
        Ok(VulkanDynamicLossScaleSeed {
            scale: self.scale,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
            growth_tracker: self.growth_tracker,
            pending_gradients_scaled: true,
        })
    }
}

struct Parameter {
    values: GpuBuffer,
    grad: GpuBuffer,
    avg: GpuBuffer,
    sq: GpuBuffer,
    len: usize,
    decay: bool,
}

pub(super) struct LossScaler {
    device: VulkanDevice,
    state: TransformerLossScaleState,
    controller: VulkanDynamicLossScaleController,
    detector: VulkanGradientNonfiniteDetector,
    adamw: vulkan::ComputeKernel,
    parameters: Vec<Parameter>,
    shared: Vec<SharedLmHeadParameter>,
    finite_checks: Vec<(GpuBuffer, usize)>,
    accumulated: BTreeMap<usize, GpuBuffer>,
    active_parameters: BTreeSet<usize>,
    pending_update: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ControlledAdamWPush {
    len: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    gradient_scale: f32,
    is_active: u32,
    apply_control_unscale: u32,
}

impl VulkanTransformer {
    pub fn supports_loss_scaler(&self) -> Result<bool> {
        Ok(self.kernels.wmma_mode != WmmaMode::Off
            && matches!(
                self.config.architecture,
                VulkanTransformerArchitecture::Gpt2 | VulkanTransformerArchitecture::Llama
            )
            && self
                .kernels
                .cpu_adamw_policy
                .lock()
                .map_err(|_| anyhow!("CPU AdamW policy lock poisoned"))?
                .is_none()
            && !self.lora_config.as_ref().is_some_and(|c| {
                !c.bias.eq_ignore_ascii_case("none")
                    || c.trainable_token_indices.is_some()
                    || c.modules_to_save.as_ref().is_some_and(|v| !v.is_empty())
                    || c.layer_replication.is_some()
                    || c.target_parameters.as_ref().is_some_and(|v| !v.is_empty())
            }))
    }

    /// Enable/restore the scaler only for the audited standalone GPT2/Llama
    /// graph. Composed image/encoder transactions need a joint overflow boundary.
    pub fn set_loss_scaler(&mut self, state: Option<TransformerLossScaleState>) -> Result<()> {
        if let Some(state) = state {
            state.seed()?;
            if !self.supports_loss_scaler()? {
                bail!("automatic loss scaling requires standalone GPT2/Llama full training or ordinary LoRA, WMMA, and native AdamW");
            }
            let scaler = LossScaler::new(&self.device, state)?;
            *self
                .kernels
                .loss_scaler
                .lock()
                .map_err(|_| anyhow!("loss scaler lock poisoned"))? = Some(scaler);
        } else {
            *self
                .kernels
                .loss_scaler
                .lock()
                .map_err(|_| anyhow!("loss scaler lock poisoned"))? = None;
        }
        Ok(())
    }

    pub fn loss_scaler_state(&self) -> Result<Option<TransformerLossScaleState>> {
        Ok(self
            .kernels
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?
            .as_ref()
            .map(|s| s.state))
    }

    /// Average complete, equal-weight microbatches before one optimizer update.
    /// Dropout is excluded until separate microbatch RNG addressing is qualified.
    pub fn set_gradient_accumulation_steps(&mut self, steps: u32) -> Result<()> {
        if !(1..=1024).contains(&steps) {
            bail!("gradient accumulation steps must be in 1..=1024");
        }
        if steps > 1
            && (self.config.attention_dropout != 0.0
                || self.config.attention_output_dropout != 0.0
                || self.config.mlp_output_dropout != 0.0
                || self.config.activation_dropout != 0.0
                || self.config.embedding_dropout != 0.0
                || self.layerdrop != 0.0
                || self
                    .lora_config
                    .as_ref()
                    .is_some_and(|c| c.lora_dropout != 0.0))
        {
            bail!(
                "precision gradient accumulation currently requires zero model and adapter dropout"
            );
        }
        let mut guard = self
            .kernels
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?;
        let Some(s) = guard.as_mut() else {
            if steps == 1 {
                return Ok(());
            }
            bail!("gradient accumulation requires an enabled WMMA loss scaler");
        };
        if s.state.pending_microbatches != 0 {
            bail!("cannot change accumulation policy inside an unfinished window");
        }
        s.state.accumulation_steps = steps;
        s.accumulated.clear();
        s.active_parameters.clear();
        Ok(())
    }
}

impl LossScaler {
    fn new(device: &VulkanDevice, state: TransformerLossScaleState) -> Result<Self> {
        state.seed()?;
        if state.pending_microbatches != 0 {
            bail!("cannot restore an unfinished accumulation window without its gradients");
        }
        Ok(Self {
            device: device.clone(),
            state,
            controller: VulkanDynamicLossScaleController::new(device)?,
            detector: VulkanGradientNonfiniteDetector::new(device.clone())?,
            adamw: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/adamw_controlled.spv"),
                5,
                std::mem::size_of::<ControlledAdamWPush>() as u32,
            )?,
            parameters: Vec::new(),
            shared: Vec::new(),
            finite_checks: Vec::new(),
            accumulated: BTreeMap::new(),
            active_parameters: BTreeSet::new(),
            pending_update: false,
        })
    }
}

impl TransformerKernels {
    pub(super) fn amp_begin(&self) -> Result<()> {
        if let Some(s) = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?
            .as_mut()
        {
            s.parameters.clear();
            s.shared.clear();
            s.finite_checks.clear();
            s.pending_update = false;
        }
        Ok(())
    }

    pub(super) fn amp_queue(&self, p: &TrainableParameter) -> Result<bool> {
        let mut guard = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?;
        let Some(s) = guard.as_mut() else {
            return Ok(false);
        };
        if !s
            .parameters
            .iter()
            .any(|q| q.values.allocation_id() == p.values.allocation_id())
        {
            s.parameters.push(Parameter {
                values: p.values.clone(),
                grad: p.grad_buffer()?.clone(),
                avg: p
                    .exp_avg
                    .as_ref()
                    .context("missing AMP first moment")?
                    .clone(),
                sq: p
                    .exp_avg_sq
                    .as_ref()
                    .context("missing AMP second moment")?
                    .clone(),
                len: p.len,
                decay: p.decay,
            });
        }
        Ok(true)
    }

    pub(super) fn amp_queue_shared(&self, p: &SharedLmHeadParameter) -> Result<bool> {
        let mut guard = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?;
        let Some(s) = guard.as_mut() else {
            return Ok(false);
        };
        let (avg, sq) = p.peft_optimizer_buffers()?;
        s.parameters.push(Parameter {
            values: p.weight_buffer()?.clone(),
            grad: p.gradient_buffer()?,
            avg,
            sq,
            len: p.context_dim() * p.vocab_size(),
            decay: true,
        });
        s.shared.push(p.clone());
        Ok(true)
    }

    pub(super) fn amp_scale_seed(
        &self,
        commands: &mut vulkan::ComputeBatch,
        seed: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        let guard = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?;
        if let Some(s) = guard.as_ref() {
            self.scale_copy.record_dispatch(
                commands,
                &[seed, seed],
                bytemuck::bytes_of(&ScalePush {
                    len: len as u32,
                    scale: s.state.scale,
                }),
                [div_ceil_u32(len, 256), 1, 1],
            )?;
        }
        Ok(())
    }

    pub(super) fn amp_watch_finite(&self, buffer: &GpuBuffer, len: usize) -> Result<()> {
        if let Some(s) = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?
            .as_mut()
        {
            s.finite_checks.push((buffer.clone(), len));
        }
        Ok(())
    }

    pub(super) fn amp_resolve(
        &self,
        commands: &mut vulkan::ComputeBatch,
        step: u32,
        hyper: AdamWHyperParams,
    ) -> Result<()> {
        let mut guard = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?;
        let Some(s) = guard.as_mut() else {
            return Ok(());
        };
        if s.parameters.is_empty() {
            bail!("AMP optimizer has no parameters");
        }
        if s.state.accumulation_steps > 1 {
            let keys = s
                .parameters
                .iter()
                .map(|p| p.values.allocation_id())
                .collect::<BTreeSet<_>>();
            if s.state.pending_microbatches > 0 && keys != s.active_parameters {
                bail!("trainable parameters changed inside a gradient accumulation window");
            }
            s.active_parameters = keys;
        }
        // Materialize unscaled FP32 gradients before the finite scan. This also
        // detects reciprocal overflow when a backed-off scale is below one,
        // and preserves the public gradient inspection ABI.
        for p in &s.parameters {
            self.scale_copy.record_dispatch(
                commands,
                &[&p.grad, &p.grad],
                bytemuck::bytes_of(&ScalePush {
                    len: p.len as u32,
                    scale: s.state.scale.recip(),
                }),
                [div_ceil_u32(p.len, 256), 1, 1],
            )?;
        }
        if s.state.accumulation_steps > 1 {
            for p in &s.parameters {
                let key = p.values.allocation_id();
                if !s.accumulated.contains_key(&key) {
                    s.accumulated
                        .insert(key, GpuBuffer::zeros_f32(&s.device, p.len)?);
                }
                let accumulated = &s.accumulated[&key];
                record_add(commands, self, &p.grad, accumulated, accumulated, p.len)?;
            }
            s.state.pending_microbatches += 1;
            if s.state.pending_microbatches < s.state.accumulation_steps {
                s.pending_update = true;
                return Ok(());
            }
            for p in &s.parameters {
                let accumulated = &s.accumulated[&p.values.allocation_id()];
                self.scale_copy.record_dispatch(
                    commands,
                    &[accumulated, &p.grad],
                    bytemuck::bytes_of(&ScalePush {
                        len: p.len as u32,
                        scale: 1.0 / s.state.accumulation_steps as f32,
                    }),
                    [div_ceil_u32(p.len, 256), 1, 1],
                )?;
                commands.fill_zero_f32(accumulated, p.len)?;
            }
            s.state.pending_microbatches = 0;
        }
        let mut gradients = s
            .parameters
            .iter()
            .map(|p| (&p.grad, p.len))
            .collect::<Vec<_>>();
        gradients.extend(s.finite_checks.iter().map(|(buffer, len)| (buffer, *len)));
        if gradients.is_empty() {
            bail!("AMP optimizer has no parameters");
        }
        s.detector.record_scan_device_only(commands, &gradients)?;
        s.controller
            .record_resolve(commands, s.detector.flag_buffer(), s.state.seed()?)?;
        for p in &s.parameters {
            let push = ControlledAdamWPush {
                len: p.len as u32,
                step,
                lr: hyper.lr,
                beta1: hyper.beta1,
                beta2: hyper.beta2,
                eps: hyper.eps,
                weight_decay: if p.decay { hyper.weight_decay } else { 0.0 },
                gradient_scale: 1.0,
                is_active: 1,
                apply_control_unscale: 0,
            };
            s.adamw.record_dispatch(
                commands,
                &[
                    &p.values,
                    &p.grad,
                    &p.avg,
                    &p.sq,
                    s.controller.control_buffer(),
                ],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(p.len, 256), 1, 1],
            )?;
        }
        s.controller.record_readback(commands)
    }

    pub(super) fn amp_unscale_output(
        &self,
        commands: &mut vulkan::ComputeBatch,
        buffer: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        let guard = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?;
        if let Some(s) = guard.as_ref() {
            self.scale_copy.record_dispatch(
                commands,
                &[buffer, buffer],
                bytemuck::bytes_of(&ScalePush {
                    len: len as u32,
                    scale: s.state.scale.recip(),
                }),
                [div_ceil_u32(len, 256), 1, 1],
            )?;
        }
        Ok(())
    }

    pub(super) fn amp_finish(&self, step: u32) -> Result<bool> {
        let mut guard = self
            .loss_scaler
            .lock()
            .map_err(|_| anyhow!("loss scaler lock poisoned"))?;
        let Some(s) = guard.as_mut() else {
            return Ok(true);
        };
        if s.pending_update {
            s.parameters.clear();
            s.shared.clear();
            s.finite_checks.clear();
            return Ok(false);
        }
        let decision = s.controller.read_decision()?;
        s.state.scale = decision.scale_after;
        s.state.growth_tracker = decision.growth_tracker;
        if decision.should_step {
            for p in &s.shared {
                p.restore_peft_optimizer_step(step);
            }
        } else {
            s.state.skipped_steps += 1;
        }
        s.parameters.clear();
        s.shared.clear();
        s.finite_checks.clear();
        s.active_parameters.clear();
        Ok(decision.should_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_microbatch_windows_match_independent_average_updates_and_skip_overflow(
    ) -> Result<()> {
        let _env = crate::test_support::env_lock();
        let device = VulkanDevice::new()?;
        let kernels = TransformerKernels::new(&device)?;
        let reference = TransformerKernels::new(&device)?;
        let state = TransformerLossScaleState {
            scale: 8.0,
            accumulation_steps: 2,
            ..Default::default()
        };
        *kernels.loss_scaler.lock().unwrap() = Some(LossScaler::new(&device, state)?);
        let p = TrainableParameter::new(&device, &[1.0], true)?;
        let expected = TrainableParameter::new(&device, &[1.0], true)?;
        let hyper = AdamWHyperParams {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 0.001,
            weight_decay: 0.1,
        };
        let mut successful_step = 0;
        for (window, values) in [[2.0f32, 6.0], [f32::INFINITY, 1.0], [1.0, 3.0]]
            .into_iter()
            .enumerate()
        {
            let next_step = successful_step + 1;
            for (micro, value) in values.into_iter().enumerate() {
                let before = p.read()?;
                kernels.amp_begin()?;
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                commands.upload_u32(p.grad_buffer()?, &[value.to_bits()])?;
                kernels.amp_scale_seed(&mut commands, p.grad_buffer()?, 1)?;
                p.record_step(&mut commands, &kernels, next_step, hyper)?;
                kernels.amp_resolve(&mut commands, next_step, hyper)?;
                commands.submit()?;
                let stepped = kernels.amp_finish(next_step)?;
                assert_eq!(stepped, micro == 1 && window != 1);
                if !stepped {
                    assert_eq!(p.read()?, before);
                }
                if micro == 0 {
                    let pending = kernels.loss_scaler.lock().unwrap().as_ref().unwrap().state;
                    assert_eq!(pending.pending_microbatches, 1);
                    assert!(
                        LossScaler::new(&device, pending).is_err(),
                        "partial gradients are required for partial resume"
                    );
                }
            }
            if window != 1 {
                successful_step += 1;
                // Analytic averages are 4 and 2; reference performs one ordinary
                // AdamW update, independent of the accumulation implementation.
                expected
                    .grad_buffer()?
                    .write_f32(&[if window == 0 { 4.0 } else { 2.0 }])?;
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                expected.record_step(&mut commands, &reference, successful_step, hyper)?;
                commands.submit()?;
            }
            assert_eq!(p.read()?, expected.read()?);
            assert_eq!(
                p.exp_avg.as_ref().unwrap().read_f32(1)?,
                expected.exp_avg.as_ref().unwrap().read_f32(1)?
            );
            assert_eq!(
                p.exp_avg_sq.as_ref().unwrap().read_f32(1)?,
                expected.exp_avg_sq.as_ref().unwrap().read_f32(1)?
            );
        }
        let state = kernels.loss_scaler.lock().unwrap().as_ref().unwrap().state;
        assert_eq!(state.pending_microbatches, 0);
        assert_eq!(state.growth_tracker, 1);
        assert_eq!(state.skipped_steps, 1);
        assert_eq!(state.scale, 4.0);
        Ok(())
    }
    #[test]
    fn global_overflow_skips_parameters_moments_and_shared_step_and_resumes_scale() -> Result<()> {
        let _env = crate::test_support::env_lock();
        let device = VulkanDevice::new()?;
        let kernels = TransformerKernels::new(&device)?;
        let state = TransformerLossScaleState {
            scale: 128.0,
            growth_interval: 2,
            ..Default::default()
        };
        *kernels.loss_scaler.lock().unwrap() = Some(LossScaler::new(&device, state)?);
        let p = TrainableParameter::new(&device, &[1.0, -1.0], true)?;
        let sibling = TrainableParameter::new(&device, &[0.5], true)?;
        let shared = SharedLmHeadParameter::new(device.clone(), 1, 1, &[0.25])?;
        let hyper = AdamWHyperParams {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 0.001,
            weight_decay: 0.1,
        };
        for (window, overflow) in [false, true, false, false].into_iter().enumerate() {
            kernels.amp_begin()?;
            let mut commands = vulkan::ComputeBatch::new(&device)?;
            commands.upload_f32(p.grad_buffer()?, &[0.25, -0.5])?;
            // Inject a raw device test value; ordinary host tensor uploads
            // deliberately reject non-finite user inputs.
            commands.upload_u32(
                sibling.grad_buffer()?,
                &[if overflow {
                    f32::INFINITY.to_bits()
                } else {
                    0.125f32.to_bits()
                }],
            )?;
            commands.upload_f32(&shared.gradient_buffer()?, &[0.0625])?;
            kernels.amp_scale_seed(&mut commands, p.grad_buffer()?, 2)?;
            kernels.amp_scale_seed(&mut commands, sibling.grad_buffer()?, 1)?;
            kernels.amp_scale_seed(&mut commands, &shared.gradient_buffer()?, 1)?;
            let step = if window < 2 {
                1 + window as u32
            } else {
                window as u32
            };
            p.record_step(&mut commands, &kernels, step, hyper)?;
            sibling.record_step(&mut commands, &kernels, step, hyper)?;
            assert!(kernels.amp_queue_shared(&shared)?);
            let before = (
                p.read()?,
                sibling.read()?,
                shared.weights()?,
                p.exp_avg.as_ref().unwrap().read_f32(2)?,
                p.exp_avg_sq.as_ref().unwrap().read_f32(2)?,
                shared.step(),
            );
            kernels.amp_resolve(&mut commands, step, hyper)?;
            commands.submit()?;
            assert_eq!(kernels.amp_finish(step)?, !overflow);
            if overflow {
                assert_eq!(
                    before,
                    (
                        p.read()?,
                        sibling.read()?,
                        shared.weights()?,
                        p.exp_avg.as_ref().unwrap().read_f32(2)?,
                        p.exp_avg_sq.as_ref().unwrap().read_f32(2)?,
                        shared.step()
                    )
                );
                assert_eq!(p.grad_buffer()?.read_f32(2)?, vec![0.0; 2]);
            } else {
                assert_ne!(before.0, p.read()?);
                assert_eq!(shared.step(), step);
            }
            let saved = kernels.loss_scaler.lock().unwrap().as_ref().unwrap().state;
            assert_eq!(
                saved.scale,
                if window == 0 || window == 3 {
                    128.0
                } else {
                    64.0
                }
            );
            assert_eq!(
                saved.growth_tracker,
                if window == 1 || window == 3 { 0 } else { 1 }
            );
            // Portable JSON resume replaces the controller, not just its host mirror.
            let restored = serde_json::from_slice(&serde_json::to_vec(&saved)?)?;
            *kernels.loss_scaler.lock().unwrap() = Some(LossScaler::new(&device, restored)?);
        }
        let avg = p.exp_avg.as_ref().unwrap().read_f32(2)?;
        assert!(
            (avg[0] - 0.06775).abs() < 2e-7,
            "gradients must be unscaled before AdamW: {avg:?}"
        );
        Ok(())
    }

    #[test]
    fn scaling_preserves_a_gradient_that_underflows_in_fp16_storage() -> Result<()> {
        let _env = crate::test_support::env_lock();
        let device = VulkanDevice::new()?;
        let kernels = TransformerKernels::new(&device)?;
        *kernels.loss_scaler.lock().unwrap() = Some(LossScaler::new(&device, Default::default())?);
        let tiny = 2f32.powi(-28);
        let raw = GpuBuffer::from_f32(&device, &[tiny, 0.0])?;
        let scaled = GpuBuffer::from_f32(&device, &[tiny, 0.0])?;
        let raw_half = GpuBuffer::zeros_u32(&device, 1)?;
        let scaled_half = GpuBuffer::zeros_u32(&device, 1)?;
        let convert = crate::wmma::WmmaGemm::new(&device, WmmaMode::Off)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernels.amp_scale_seed(&mut commands, &scaled, 2)?;
        convert.record_conversion(&mut commands, &raw, &raw_half, 2)?;
        convert.record_conversion(&mut commands, &scaled, &scaled_half, 2)?;
        commands.submit()?;
        assert_eq!(raw_half.read_u32(1)?[0] & 0xffff, 0);
        let preserved = crate::wmma::f16_bits_to_f32(scaled_half.read_u32(1)?[0] as u16);
        assert_eq!(preserved / 256.0, tiny);
        Ok(())
    }
}
