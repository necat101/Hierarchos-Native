use std::collections::HashSet;

use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: 0,
        }
    }
}

pub struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            rng: StdRng::seed_from_u64(config.seed),
            config,
        }
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        if self.config.repetition_penalty > 0.0
            && (self.config.repetition_penalty - 1.0).abs() > f32::EPSILON
        {
            let unique_history: HashSet<u32> = history.iter().copied().collect();
            for token in unique_history {
                if let Some((_, logit)) = candidates.get_mut(token as usize) {
                    if *logit < 0.0 {
                        *logit *= self.config.repetition_penalty;
                    } else {
                        *logit /= self.config.repetition_penalty;
                    }
                }
            }
        }
        if self.config.temperature <= 0.0 {
            return candidates
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(idx, _)| *idx as u32)
                .unwrap_or(0);
        }
        let inv_temp = 1.0 / self.config.temperature.max(1e-6);
        for (_, logit) in &mut candidates {
            *logit *= inv_temp;
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if self.config.top_k > 0 && candidates.len() > self.config.top_k {
            candidates.truncate(self.config.top_k);
        }

        let max_logit = candidates.first().map(|v| v.1).unwrap_or(0.0);
        let mut probs: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, logit)| (idx, (logit - max_logit).exp()))
            .collect();
        let total = probs
            .iter()
            .map(|v| v.1)
            .sum::<f32>()
            .max(f32::MIN_POSITIVE);
        for (_, p) in &mut probs {
            *p /= total;
        }

        if self.config.top_p > 0.0 && self.config.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut keep = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.config.top_p {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep.max(1));
            let renorm = probs
                .iter()
                .map(|v| v.1)
                .sum::<f32>()
                .max(f32::MIN_POSITIVE);
            for (_, p) in &mut probs {
                *p /= renorm;
            }
        }

        let target = self.rng.random::<f32>();
        let mut cumulative = 0.0f32;
        for &(idx, p) in &probs {
            cumulative += p;
            if target <= cumulative {
                return idx as u32;
            }
        }
        probs.last().map(|v| v.0 as u32).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy_with_penalty(penalty: f32) -> Sampler {
        Sampler::new(SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: penalty,
            seed: 0,
        })
    }

    #[test]
    fn greedy_decoding_applies_repetition_penalty() {
        let mut sampler = greedy_with_penalty(2.0);
        assert_eq!(sampler.sample(&[1.0, 0.9], &[0]), 1);
    }

    #[test]
    fn repeated_history_token_is_penalized_only_once() {
        let mut sampler = greedy_with_penalty(2.0);
        assert_eq!(sampler.sample(&[1.0, 0.4], &[0, 0, 0]), 0);
    }
}
