use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RosaTransitionSnapshot {
    pub symbol: u32,
    pub target: usize,
}

/// Backend-neutral ROSA automaton snapshot. Transitions are stored as sorted
/// rows rather than serialized hash maps so JSON is deterministic and trivial
/// to reconstruct in PyTorch, Vulkan, Rust, or another runtime.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RosaStateSnapshot {
    pub transitions: Vec<Vec<RosaTransitionSnapshot>>,
    pub suffix_links: Vec<isize>,
    pub lengths: Vec<usize>,
    pub endpos: Vec<isize>,
    pub last_state: usize,
    pub tokens: Vec<u32>,
}

/// Incremental Rapid Online Suffix Automaton state.
///
/// This is a direct state-machine port of `hierarchos.utils.rosa._rosa_incremental`.
#[derive(Clone, Debug)]
pub struct RosaState {
    transitions: Vec<HashMap<u32, usize>>,
    suffix_links: Vec<isize>,
    lengths: Vec<usize>,
    endpos: Vec<isize>,
    last_state: usize,
    tokens: Vec<u32>,
}

impl Default for RosaState {
    fn default() -> Self {
        Self::new()
    }
}

impl RosaState {
    pub fn new() -> Self {
        Self {
            transitions: vec![HashMap::new()],
            suffix_links: vec![-1],
            lengths: vec![0],
            endpos: vec![-1],
            last_state: 0,
            tokens: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn snapshot(&self) -> RosaStateSnapshot {
        let transitions = self
            .transitions
            .iter()
            .map(|row| {
                let mut row = row
                    .iter()
                    .map(|(&symbol, &target)| RosaTransitionSnapshot { symbol, target })
                    .collect::<Vec<_>>();
                row.sort_by_key(|transition| transition.symbol);
                row
            })
            .collect();
        RosaStateSnapshot {
            transitions,
            suffix_links: self.suffix_links.clone(),
            lengths: self.lengths.clone(),
            endpos: self.endpos.clone(),
            last_state: self.last_state,
            tokens: self.tokens.clone(),
        }
    }

    pub fn from_snapshot(snapshot: &RosaStateSnapshot) -> Result<Self> {
        let states = snapshot.transitions.len();
        if states == 0
            || snapshot.suffix_links.len() != states
            || snapshot.lengths.len() != states
            || snapshot.endpos.len() != states
            || snapshot.last_state >= states
        {
            return Err(Error::Invalid(
                "ROSA runtime-state snapshot has inconsistent state-table geometry".into(),
            ));
        }
        if snapshot.suffix_links[0] != -1 || snapshot.lengths[0] != 0 {
            return Err(Error::Invalid(
                "ROSA runtime-state snapshot has an invalid root state".into(),
            ));
        }
        if snapshot.lengths[snapshot.last_state] != snapshot.tokens.len() {
            return Err(Error::Invalid(
                "ROSA runtime-state final state does not cover its token history".into(),
            ));
        }

        let token_count = snapshot.tokens.len();
        let mut transitions = Vec::with_capacity(states);
        for (source, row) in snapshot.transitions.iter().enumerate() {
            let mut restored = HashMap::with_capacity(row.len());
            for transition in row {
                if transition.target >= states
                    || snapshot.lengths[transition.target] <= snapshot.lengths[source]
                    || restored
                        .insert(transition.symbol, transition.target)
                        .is_some()
                {
                    return Err(Error::Invalid(
                        "ROSA runtime-state snapshot contains an invalid transition".into(),
                    ));
                }
            }
            transitions.push(restored);
        }
        for state in 0..states {
            let suffix = snapshot.suffix_links[state];
            if state == 0 {
                if suffix != -1 {
                    return Err(Error::Invalid(
                        "ROSA runtime-state root suffix link must be -1".into(),
                    ));
                }
            } else if suffix < 0
                || suffix as usize >= states
                || snapshot.lengths[suffix as usize] >= snapshot.lengths[state]
            {
                return Err(Error::Invalid(
                    "ROSA runtime-state snapshot contains an invalid suffix link".into(),
                ));
            }
            let end = snapshot.endpos[state];
            if end < -1 || end >= token_count as isize {
                return Err(Error::Invalid(
                    "ROSA runtime-state snapshot contains an invalid end position".into(),
                ));
            }
        }

        Ok(Self {
            transitions,
            suffix_links: snapshot.suffix_links.clone(),
            lengths: snapshot.lengths.clone(),
            endpos: snapshot.endpos.clone(),
            last_state: snapshot.last_state,
            tokens: snapshot.tokens.clone(),
        })
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn push_state(&mut self) -> usize {
        let idx = self.transitions.len();
        self.transitions.push(HashMap::new());
        self.suffix_links.push(-1);
        self.lengths.push(0);
        self.endpos.push(-1);
        idx
    }

    /// Consume one token and return the predicted next-token ID, or `None`.
    pub fn predict_and_push(&mut self, token: u32, max_context: usize) -> Option<u32> {
        if max_context > 0 && self.tokens.len() >= max_context {
            self.reset();
        }

        let i = self.tokens.len();
        self.tokens.push(token);

        let previous_last = self.last_state;
        let r = self.push_state();
        self.lengths[r] = self.lengths[previous_last] + 1;

        let mut p = previous_last as isize;
        while p != -1 && !self.transitions[p as usize].contains_key(&token) {
            self.transitions[p as usize].insert(token, r);
            p = self.suffix_links[p as usize];
        }

        if p == -1 {
            self.suffix_links[r] = 0;
        } else {
            let p_idx = p as usize;
            let q = self.transitions[p_idx][&token];
            if self.lengths[p_idx] + 1 == self.lengths[q] {
                self.suffix_links[r] = q as isize;
            } else {
                let u = self.push_state();
                self.transitions[u] = self.transitions[q].clone();
                self.lengths[u] = self.lengths[p_idx] + 1;
                self.suffix_links[u] = self.suffix_links[q];
                self.endpos[u] = self.endpos[q];

                let mut walk = p;
                while walk != -1 {
                    let walk_idx = walk as usize;
                    if self.transitions[walk_idx].get(&token).copied() != Some(q) {
                        break;
                    }
                    self.transitions[walk_idx].insert(token, u);
                    walk = self.suffix_links[walk_idx];
                }
                self.suffix_links[q] = u as isize;
                self.suffix_links[r] = u as isize;
            }
        }

        self.last_state = r;
        let mut v = r as isize;
        let mut prediction = None;
        while v != -1 {
            let idx = v as usize;
            if self.lengths[idx] > 0 && self.endpos[idx] >= 0 {
                let next_pos = self.endpos[idx] as usize + 1;
                if next_pos < self.tokens.len() {
                    prediction = Some(self.tokens[next_pos]);
                }
                break;
            }
            v = self.suffix_links[idx];
        }

        let mut v = r as isize;
        while v != -1 {
            let idx = v as usize;
            if self.endpos[idx] >= i as isize {
                break;
            }
            self.endpos[idx] = i as isize;
            v = self.suffix_links[idx];
        }

        prediction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suffix_match_reference(tokens: &[u32], max_context: usize) -> Vec<Option<u32>> {
        assert!(max_context > 0);
        let mut history = Vec::with_capacity(max_context);
        let mut match_state = vec![0usize; max_context * 2];
        let mut predictions = Vec::with_capacity(tokens.len());

        for &token in tokens {
            if history.len() >= max_context {
                history.clear();
            }
            history.push(token);
            let current = history.len() - 1;
            if current == 0 {
                predictions.push(None);
                continue;
            }

            let dst_generation = current & 1;
            let src_generation = dst_generation ^ 1;
            let dst_base = dst_generation * max_context;
            let src_base = src_generation * max_context;
            let mut best_match = 0usize;
            let mut best_following = 0usize;

            for prior_end in 0..current {
                let match_len = if history[current] == history[prior_end] {
                    1 + if prior_end == 0 {
                        0
                    } else {
                        match_state[src_base + prior_end - 1]
                    }
                } else {
                    0
                };
                match_state[dst_base + prior_end] = match_len;
                if match_len > best_match
                    || (match_len == best_match && match_len != 0 && prior_end + 1 > best_following)
                {
                    best_match = match_len;
                    best_following = prior_end + 1;
                }
            }

            predictions.push((best_match != 0).then(|| history[best_following]));
        }

        predictions
    }

    #[test]
    fn repeated_pattern_predicts_next_token() {
        let mut state = RosaState::new();
        let seq = [1, 2, 1, 2, 1];
        let predictions: Vec<_> = seq
            .into_iter()
            .map(|t| state.predict_and_push(t, 0))
            .collect();
        assert_eq!(predictions[0], None);
        assert_eq!(predictions[1], None);
        assert_eq!(predictions[2], Some(2));
        assert_eq!(predictions[3], Some(1));
        assert_eq!(predictions[4], Some(2));
    }

    #[test]
    fn bounded_mode_resets_on_segment_boundary() {
        let mut state = RosaState::new();
        for t in [1, 2, 1, 2] {
            state.predict_and_push(t, 4);
        }
        assert_eq!(state.len(), 4);
        assert_eq!(state.predict_and_push(9, 4), None);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn incremental_suffix_match_state_is_exactly_rosa_equivalent() {
        let mut tokens = vec![1, 2, 1, 2, 1, 3, 1, 2, 1, 2, 4, 4, 4, 4];
        let mut lcg = 0x2026_0814u32;
        for _ in 0..512 {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            tokens.push(1 + ((lcg >> 16) % 11));
        }

        for max_context in 1..=32 {
            let mut state = RosaState::new();
            let expected = tokens
                .iter()
                .copied()
                .map(|token| state.predict_and_push(token, max_context))
                .collect::<Vec<_>>();
            let actual = suffix_match_reference(&tokens, max_context);
            assert_eq!(
                actual, expected,
                "two-generation suffix state drifted at max_context={max_context}"
            );
        }
    }

    #[test]
    fn snapshot_roundtrip_preserves_incremental_predictions() {
        let mut state = RosaState::new();
        for token in [1, 2, 1, 2, 3, 1, 2, 1] {
            state.predict_and_push(token, 0);
        }
        let encoded = serde_json::to_string(&state.snapshot()).expect("serialize ROSA snapshot");
        let decoded: RosaStateSnapshot =
            serde_json::from_str(&encoded).expect("deserialize ROSA snapshot");
        let mut restored = RosaState::from_snapshot(&decoded).expect("restore ROSA snapshot");

        for token in [2, 4, 1, 2, 1, 2] {
            assert_eq!(
                state.predict_and_push(token, 0),
                restored.predict_and_push(token, 0)
            );
        }
        assert_eq!(state.snapshot(), restored.snapshot());
    }
}
