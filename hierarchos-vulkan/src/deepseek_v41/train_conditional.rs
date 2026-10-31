use super::*;

/// Optional gradients for the two independent LoRA parameters. None skips
/// that parameter's decay, moments and counter; Some(zeros) is an update.
#[derive(Debug, Clone)]
pub struct DeepseekV41AdapterParameterGradient {
    pub target: String,
    pub grad_a: Option<Vec<f32>>,
    pub grad_b: Option<Vec<f32>>,
}

struct GradientRef<'a> {
    target: &'a str,
    a: Option<&'a [f32]>,
    b: Option<&'a [f32]>,
}
struct Prepared<'a> {
    target: &'a str,
    is_a: bool,
    gradient: &'a [f32],
    step: u64,
    step_size: f32,
    bias2_sqrt: f32,
}
impl DeepseekV41AdapterTrainer {
    /// AdamW over active projections. Omitted targets match PyTorch grad=None
    /// and remain completely unchanged. An empty set advances only call count.
    pub fn step(&mut self, gradients: &[DeepseekV41AdapterGradient]) -> Result<()> {
        self.step_refs(
            &gradients
                .iter()
                .map(|g| GradientRef {
                    target: &g.target,
                    a: Some(&g.grad_a),
                    b: Some(&g.grad_b),
                })
                .collect::<Vec<_>>(),
        )
    }

    /// Require every adapted projection, for callers that expect a dense graph.
    pub fn step_complete(&mut self, gradients: &[DeepseekV41AdapterGradient]) -> Result<()> {
        if gradients.len() != self.factors.len() {
            bail!(
                "V4.1 complete adapter step received {} gradients for {} targets",
                gradients.len(),
                self.factors.len()
            );
        }
        self.step(gradients)
    }

    /// Support independently absent A/B gradients as well as omitted modules.
    pub fn step_parameters(
        &mut self,
        gradients: &[DeepseekV41AdapterParameterGradient],
    ) -> Result<()> {
        self.step_refs(
            &gradients
                .iter()
                .map(|g| GradientRef {
                    target: &g.target,
                    a: g.grad_a.as_deref(),
                    b: g.grad_b.as_deref(),
                })
                .collect::<Vec<_>>(),
        )
    }

    /// Per-parameter AdamW update counts (A,B), independent of optimizer calls.
    pub fn parameter_step_counts(&self, target: &str) -> Option<(u64, u64)> {
        self.slots.get(target).map(|s| (s.step_a, s.step_b))
    }

    fn step_refs(&mut self, gradients: &[GradientRef<'_>]) -> Result<()> {
        let calls = self
            .step
            .checked_add(1)
            .context("V4.1 optimizer call count overflow")?;
        let mut seen = std::collections::BTreeSet::new();
        let mut prepared = Vec::new();
        let legacy=self.math_policy==AdamWMathPolicy::LegacyF32V1;
        let lr = if legacy {(self.config.learning_rate as f32) as f64} else {self.config.learning_rate};
        let wd=if legacy {(self.config.weight_decay as f32) as f64} else {self.config.weight_decay};
        let b1=if legacy {(self.config.beta1 as f32) as f64} else {self.config.beta1};
        let b2=if legacy {(self.config.beta2 as f32) as f64} else {self.config.beta2};
        let decay = (1.0 - lr * wd) as f32;
        for g in gradients {
            if !seen.insert(g.target) {
                bail!(
                    "V4.1 adapter step received duplicate gradients for {}",
                    g.target
                );
            }
            let factors = self.factors.get(g.target).with_context(|| {
                format!("V4.1 adapter gradient has unknown target {}", g.target)
            })?;
            let slot = self
                .slots
                .get(g.target)
                .with_context(|| format!("V4.1 optimizer slot for {} is missing", g.target))?;
            for (is_a, gradient, param, avg, square, old_step) in [
                (
                    true,
                    g.a,
                    &factors.a,
                    &slot.exp_avg_a,
                    &slot.exp_avg_sq_a,
                    slot.step_a,
                ),
                (
                    false,
                    g.b,
                    &factors.b,
                    &slot.exp_avg_b,
                    &slot.exp_avg_sq_b,
                    slot.step_b,
                ),
            ] {
                let Some(gradient) = gradient else {
                    continue;
                };
                if gradient.len() != param.len()
                    || (avg.len() != param.len() && !(old_step == 0 && avg.is_empty()))
                    || (square.len() != param.len() && !(old_step == 0 && square.is_empty()))
                    || gradient.iter().any(|v| !v.is_finite())
                {
                    bail!(
                        "V4.1 adapter gradient geometry/finiteness does not match {}",
                        g.target
                    );
                }
                let step = old_step
                    .checked_add(1)
                    .filter(|&s| s <= i32::MAX as u64)
                    .context("V4.1 adapter parameter step exceeds bias-correction range")?;
                let step_size = (lr / (1.0 - b1.powi(step as i32))) as f32;
                let bias2_sqrt = (1.0 - b2.powi(step as i32)).sqrt() as f32;
                // Dry-run every active update before changing any state. This
                // catches finite-gradient overflow without cloning factors or
                // moments, and prevents a bad late parameter's partial step.
                for i in 0..param.len() {
                    let (p, m, v) = update_element(
                        param[i],
                        gradient[i],
                        avg.get(i).copied().unwrap_or(0.0),
                        square.get(i).copied().unwrap_or(0.0),
                        self.config,
                        decay,
                        step_size,
                        bias2_sqrt,
                        self.math_policy,
                        i<param.len()/(2*crate::transformer::aten_vector_width() as usize)*(2*crate::transformer::aten_vector_width() as usize),
                    );
                    if !p.is_finite() || !m.is_finite() || !v.is_finite() || v < 0.0 {
                        bail!(
                            "V4.1 adapter update for {} produces non-finite state",
                            g.target
                        );
                    }
                }
                prepared.push(Prepared {
                    target: g.target,
                    is_a,
                    gradient,
                    step,
                    step_size,
                    bias2_sqrt,
                });
            }
        }
        for p in prepared {
            let factors = self.factors.get_mut(p.target).expect("validated target");
            let slot = self
                .slots
                .get_mut(p.target)
                .expect("validated optimizer slot");
            let (param, avg, square, counter) = if p.is_a {
                (
                    &mut factors.a,
                    &mut slot.exp_avg_a,
                    &mut slot.exp_avg_sq_a,
                    &mut slot.step_a,
                )
            } else {
                (
                    &mut factors.b,
                    &mut slot.exp_avg_b,
                    &mut slot.exp_avg_sq_b,
                    &mut slot.step_b,
                )
            };
            if *counter == 0 {
                avg.resize(param.len(), 0.0);
                square.resize(param.len(), 0.0);
            }
            update(
                param,
                p.gradient,
                avg,
                square,
                self.config,
                decay,
                p.step_size,
                p.bias2_sqrt,
                self.math_policy,
            );
            *counter = p.step;
        }
        self.step = calls;
        Ok(())
    }
}
