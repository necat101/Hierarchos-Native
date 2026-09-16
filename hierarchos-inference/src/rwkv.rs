use std::sync::Arc;

use crate::{
    error::{Error, Result},
    math::{finite_clamp_vec, group_norm, layer_norm, sigmoid, softplus, Linear, Matrix},
    weights::WeightLoader,
};

#[derive(Clone, Debug)]
pub(crate) struct RwkvState {
    pub prev_tm: Vec<f32>,
    pub prev_cm: Vec<f32>,
    pub v_first: Vec<f32>,
    pub output: Vec<f32>,
    /// Per-head matrix state, row-major [head, N, N].
    pub matrix: Vec<f32>,
}

impl RwkvState {
    pub(crate) const EXPLICIT_OUTPUT_MATRIX_OFFSET: usize = 4;

    pub fn zeros(width: usize, head_size: usize) -> Self {
        let heads = width / head_size;
        Self {
            prev_tm: vec![0.0; width],
            prev_cm: vec![0.0; width],
            v_first: vec![0.0; width],
            output: vec![0.0; width],
            matrix: vec![0.0; heads * head_size * head_size],
        }
    }

    /// Export the recurrent state in the exact coherent-v9 PyTorch/Vulkan
    /// packed layout `[B=1, C=width, 4 + head_size]`:
    ///
    /// `prev_tm, prev_cm, v_first, explicit_output, matrix_row...`.
    ///
    /// Keeping the interchange representation identical to the training
    /// backends avoids a Rust-only recurrent-state ABI and lets a live session
    /// cross backend boundaries without numerically reconstructing state.
    pub(crate) fn to_explicit_output_packed(
        &self,
        width: usize,
        head_size: usize,
    ) -> Result<Vec<f32>> {
        if head_size == 0 || width == 0 || !width.is_multiple_of(head_size) {
            return Err(Error::Invalid(format!(
                "RWKV runtime-state geometry requires positive width/head_size with width divisible by head_size; got width={width} head_size={head_size}"
            )));
        }
        let matrix_values = width
            .checked_mul(head_size)
            .ok_or_else(|| Error::Invalid("RWKV runtime-state matrix geometry overflow".into()))?;
        for (name, actual, expected) in [
            ("prev_tm", self.prev_tm.len(), width),
            ("prev_cm", self.prev_cm.len(), width),
            ("v_first", self.v_first.len(), width),
            ("output", self.output.len(), width),
            ("matrix", self.matrix.len(), matrix_values),
        ] {
            if actual != expected {
                return Err(Error::Invalid(format!(
                    "RWKV runtime-state {name} has {actual} values; expected {expected}"
                )));
            }
        }
        if self
            .prev_tm
            .iter()
            .chain(&self.prev_cm)
            .chain(&self.v_first)
            .chain(&self.output)
            .chain(&self.matrix)
            .any(|value| !value.is_finite())
        {
            return Err(Error::Invalid(
                "RWKV runtime state contains non-finite values".into(),
            ));
        }

        let state_size = Self::EXPLICIT_OUTPUT_MATRIX_OFFSET + head_size;
        let mut packed = Vec::with_capacity(width * state_size);
        for channel in 0..width {
            packed.push(self.prev_tm[channel]);
            packed.push(self.prev_cm[channel]);
            packed.push(self.v_first[channel]);
            packed.push(self.output[channel]);
            let matrix_start = channel * head_size;
            packed.extend_from_slice(&self.matrix[matrix_start..matrix_start + head_size]);
        }
        Ok(packed)
    }

    pub(crate) fn from_explicit_output_packed(
        width: usize,
        head_size: usize,
        packed: &[f32],
    ) -> Result<Self> {
        if head_size == 0 || width == 0 || !width.is_multiple_of(head_size) {
            return Err(Error::Invalid(format!(
                "RWKV runtime-state geometry requires positive width/head_size with width divisible by head_size; got width={width} head_size={head_size}"
            )));
        }
        let state_size = Self::EXPLICIT_OUTPUT_MATRIX_OFFSET + head_size;
        let expected = width
            .checked_mul(state_size)
            .ok_or_else(|| Error::Invalid("RWKV runtime-state packed geometry overflow".into()))?;
        if packed.len() != expected {
            return Err(Error::Invalid(format!(
                "RWKV packed runtime state has {} values; expected {expected} for [1,{width},{state_size}]",
                packed.len()
            )));
        }
        if packed.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid(
                "RWKV packed runtime state contains non-finite values".into(),
            ));
        }

        let mut state = Self::zeros(width, head_size);
        for channel in 0..width {
            let packed_start = channel * state_size;
            state.prev_tm[channel] = packed[packed_start];
            state.prev_cm[channel] = packed[packed_start + 1];
            state.v_first[channel] = packed[packed_start + 2];
            state.output[channel] = packed[packed_start + 3];
            let matrix_start = channel * head_size;
            state.matrix[matrix_start..matrix_start + head_size]
                .copy_from_slice(&packed[packed_start + 4..packed_start + state_size]);
        }
        Ok(state)
    }
}

#[derive(Clone)]
pub(crate) struct RwkvCell {
    width: usize,
    head_size: usize,
    heads: usize,
    state_clamp: f32,
    channel_mix_key_clamp: f32,
    channel_mix_deepembed_clamp: f32,

    x_r: Arc<[f32]>,
    x_w: Arc<[f32]>,
    x_k: Arc<[f32]>,
    x_v: Arc<[f32]>,
    x_a: Arc<[f32]>,
    x_g: Arc<[f32]>,
    w1: Matrix,
    w2: Matrix,
    w0: Arc<[f32]>,
    a1: Matrix,
    a2: Matrix,
    a0: Arc<[f32]>,
    g1: Matrix,
    g2: Matrix,
    k_k: Arc<[f32]>,
    k_a: Arc<[f32]>,
    r_k: Arc<[f32]>,
    x_k_cm: Arc<[f32]>,

    ln1_weight: Arc<[f32]>,
    ln1_bias: Arc<[f32]>,
    ln2_weight: Arc<[f32]>,
    ln2_bias: Arc<[f32]>,
    ln_x_weight: Arc<[f32]>,
    ln_x_bias: Arc<[f32]>,

    receptance: Linear,
    key: Linear,
    value: Linear,
    output_proj: Linear,
    key_cm: Linear,
    value_cm: Linear,
}

impl RwkvCell {
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        width: usize,
        head_size: usize,
        state_clamp: f32,
        channel_mix_key_clamp: f32,
        channel_mix_deepembed_clamp: f32,
    ) -> Result<Self> {
        let heads = width / head_size;
        let w1 = load_parameter_matrix(loader, &format!("{prefix}.w1"), width)?;
        let w2 = loader.matrix(&format!("{prefix}.w2"), w1.cols, width)?;
        let a1 = load_parameter_matrix(loader, &format!("{prefix}.a1"), width)?;
        let a2 = loader.matrix(&format!("{prefix}.a2"), a1.cols, width)?;
        let g1 = load_parameter_matrix(loader, &format!("{prefix}.g1"), width)?;
        let g2 = loader.matrix(&format!("{prefix}.g2"), g1.cols, width)?;

        Ok(Self {
            width,
            head_size,
            heads,
            state_clamp,
            channel_mix_key_clamp,
            channel_mix_deepembed_clamp,
            x_r: loader.vector(&format!("{prefix}.x_r"), width)?,
            x_w: loader.vector(&format!("{prefix}.x_w"), width)?,
            x_k: loader.vector(&format!("{prefix}.x_k"), width)?,
            x_v: loader.vector(&format!("{prefix}.x_v"), width)?,
            x_a: loader.vector(&format!("{prefix}.x_a"), width)?,
            x_g: loader.vector(&format!("{prefix}.x_g"), width)?,
            w1,
            w2,
            w0: loader.vector(&format!("{prefix}.w0"), width)?,
            a1,
            a2,
            a0: loader.vector(&format!("{prefix}.a0"), width)?,
            g1,
            g2,
            k_k: loader.vector(&format!("{prefix}.k_k"), width)?,
            k_a: loader.vector(&format!("{prefix}.k_a"), width)?,
            r_k: loader.flat(&format!("{prefix}.r_k"), width)?,
            x_k_cm: loader.vector(&format!("{prefix}.x_k_cm"), width)?,
            ln1_weight: loader.vector(&format!("{prefix}.ln1.weight"), width)?,
            ln1_bias: loader.vector(&format!("{prefix}.ln1.bias"), width)?,
            ln2_weight: loader.vector(&format!("{prefix}.ln2.weight"), width)?,
            ln2_bias: loader.vector(&format!("{prefix}.ln2.bias"), width)?,
            ln_x_weight: loader.vector(&format!("{prefix}.ln_x.weight"), width)?,
            ln_x_bias: loader.vector(&format!("{prefix}.ln_x.bias"), width)?,
            receptance: loader.linear(&format!("{prefix}.receptance"), width, width, false)?,
            key: loader.linear(&format!("{prefix}.key"), width, width, false)?,
            value: loader.linear(&format!("{prefix}.value"), width, width, false)?,
            output_proj: loader.linear(&format!("{prefix}.output"), width, width, false)?,
            key_cm: loader.linear(&format!("{prefix}.key_cm"), width * 4, width, false)?,
            value_cm: loader.linear(&format!("{prefix}.value_cm"), width, width * 4, false)?,
        })
    }

    pub fn zero_state(&self) -> RwkvState {
        RwkvState::zeros(self.width, self.head_size)
    }

    #[inline]
    fn mix(x: &[f32], previous: &[f32], coeff: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(previous.iter())
            .zip(coeff.iter())
            .map(|((&xv, &pv), &c)| xv + (pv - xv) * c)
            .collect()
    }

    pub fn step(
        &self,
        x: &[f32],
        state: &RwkvState,
        deepembed: Option<&[f32]>,
    ) -> (Vec<f32>, RwkvState) {
        debug_assert_eq!(x.len(), self.width);
        let residual_tm = x;
        let x_norm = layer_norm(x, Some(&self.ln1_weight), Some(&self.ln1_bias), 1e-5);

        let xr = Self::mix(&x_norm, &state.prev_tm, &self.x_r);
        let xw = Self::mix(&x_norm, &state.prev_tm, &self.x_w);
        let xk = Self::mix(&x_norm, &state.prev_tm, &self.x_k);
        let xv = Self::mix(&x_norm, &state.prev_tm, &self.x_v);
        let xa = Self::mix(&x_norm, &state.prev_tm, &self.x_a);
        let xg = Self::mix(&x_norm, &state.prev_tm, &self.x_g);

        let r = self.receptance.forward(&xr);
        let mut k = self.key.forward(&xk);
        let v = self.value.forward(&xv);
        // Hierarchos H and L both instantiate layer_id=0, so v_first == v.
        let v_first = v.clone();

        let a_hidden = self.a1.row_vec_mat(&xa);
        let a_delta = self.a2.row_vec_mat(&a_hidden);
        let a: Vec<f32> = (0..self.width)
            .map(|i| sigmoid(self.a0[i] + a_delta[i]))
            .collect();

        let g_hidden = self.g1.row_vec_mat(&xg);
        let g_sigmoid: Vec<f32> = g_hidden.into_iter().map(sigmoid).collect();
        let g = self.g2.row_vec_mat(&g_sigmoid);

        let w_hidden = self.w1.row_vec_mat(&xw);
        let w_tanh: Vec<f32> = w_hidden.into_iter().map(f32::tanh).collect();
        let w_delta = self.w2.row_vec_mat(&w_tanh);
        let w: Vec<f32> = (0..self.width)
            .map(|i| -softplus(-(self.w0[i] + w_delta[i])) - 0.5)
            .collect();

        let mut kk = vec![0.0f32; self.width];
        for h in 0..self.heads {
            let start = h * self.head_size;
            let end = start + self.head_size;
            let mut norm_sq = 0.0f32;
            for i in start..end {
                let value = k[i] * self.k_k[i];
                kk[i] = value;
                norm_sq += value * value;
            }
            // torch.nn.functional.normalize(..., eps=1e-12)
            let denom = norm_sq.sqrt().max(1e-12);
            for value in &mut kk[start..end] {
                *value /= denom;
            }
        }

        for i in 0..self.width {
            k[i] *= 1.0 + (a[i] - 1.0) * self.k_a[i];
        }

        let mut matrix = state.matrix.clone();
        let mut tmix_raw = vec![0.0f32; self.width];
        for h in 0..self.heads {
            let c0 = h * self.head_size;
            let matrix0 = h * self.head_size * self.head_size;

            let mut sa = vec![0.0f32; self.head_size];
            for (row, sa_value) in sa.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for col in 0..self.head_size {
                    let idx = matrix0 + row * self.head_size + col;
                    sum = matrix[idx].mul_add(-kk[c0 + col], sum);
                }
                *sa_value = sum;
            }

            for row in 0..self.head_size {
                for col in 0..self.head_size {
                    let idx = matrix0 + row * self.head_size + col;
                    let bounded_w = if w[c0 + col].is_finite() {
                        w[c0 + col].clamp(-60.0, 30.0)
                    } else {
                        w[c0 + col]
                    };
                    let decay = (-bounded_w.exp()).exp();
                    let b = kk[c0 + col] * a[c0 + col];
                    matrix[idx] = matrix[idx] * decay + sa[row] * b + v[c0 + row] * k[c0 + col];
                }
            }

            for row in 0..self.head_size {
                let mut sum = 0.0f32;
                for col in 0..self.head_size {
                    let idx = matrix0 + row * self.head_size + col;
                    sum = matrix[idx].mul_add(r[c0 + col], sum);
                }
                tmix_raw[c0 + row] = sum;
            }
        }

        let mut tmix = group_norm(
            &tmix_raw,
            self.heads,
            &self.ln_x_weight,
            &self.ln_x_bias,
            64e-5,
        );
        for h in 0..self.heads {
            let start = h * self.head_size;
            let end = start + self.head_size;
            let mut bonus_scale = 0.0f32;
            for i in start..end {
                bonus_scale += r[i] * k[i] * self.r_k[i];
            }
            for i in start..end {
                tmix[i] += bonus_scale * v[i];
            }
        }
        for i in 0..self.width {
            tmix[i] *= g[i];
        }
        let tmix_projected = self.output_proj.forward(&tmix);
        let mut mixed = Vec::with_capacity(self.width);
        for i in 0..self.width {
            mixed.push(residual_tm[i] + tmix_projected[i]);
        }

        let x_norm2 = layer_norm(&mixed, Some(&self.ln2_weight), Some(&self.ln2_bias), 1e-5);
        let xk_cm = Self::mix(&x_norm2, &state.prev_cm, &self.x_k_cm);
        let mut cm_key = self.key_cm.forward(&xk_cm);
        if self.channel_mix_key_clamp > 0.0 {
            finite_clamp_vec(&mut cm_key, self.channel_mix_key_clamp);
        }
        let mut ffn: Vec<f32> = cm_key
            .into_iter()
            .map(|value| value.max(0.0).powi(2))
            .collect();
        if let Some(deepembed) = deepembed {
            debug_assert_eq!(deepembed.len(), ffn.len());
            for i in 0..ffn.len() {
                let mut d = deepembed[i];
                if self.channel_mix_deepembed_clamp > 0.0 && d.is_finite() {
                    d = d.clamp(
                        -self.channel_mix_deepembed_clamp,
                        self.channel_mix_deepembed_clamp,
                    );
                }
                ffn[i] *= d;
            }
            if self.channel_mix_key_clamp > 0.0 && self.channel_mix_deepembed_clamp > 0.0 {
                let limit = self.channel_mix_key_clamp
                    * self.channel_mix_key_clamp
                    * self.channel_mix_deepembed_clamp;
                finite_clamp_vec(&mut ffn, limit);
            }
        }
        let cm_projected = self.value_cm.forward(&ffn);
        let mut output = Vec::with_capacity(self.width);
        for i in 0..self.width {
            output.push(mixed[i] + cm_projected[i]);
        }

        let mut new_state = RwkvState {
            prev_tm: x_norm,
            prev_cm: x_norm2,
            v_first,
            output: output.clone(),
            matrix,
        };
        finite_clamp_vec(&mut new_state.prev_tm, self.state_clamp);
        finite_clamp_vec(&mut new_state.prev_cm, self.state_clamp);
        finite_clamp_vec(&mut new_state.v_first, self.state_clamp);
        finite_clamp_vec(&mut new_state.output, self.state_clamp);
        finite_clamp_vec(&mut new_state.matrix, self.state_clamp);
        (output, new_state)
    }
}

fn load_parameter_matrix(loader: &WeightLoader, name: &str, rows: usize) -> Result<Matrix> {
    // The low-rank width is encoded in the tensor itself. Safetensors does not
    // expose a cheap shape-only query through WeightLoader, so infer the known
    // Hierarchos rank formula from the model width. Current H/L cells are layer 0.
    let rank = if name.ends_with(".g1") {
        rwkv_lora_rank(rows, 5.0)
    } else {
        rwkv_lora_rank(rows, 2.5)
    };
    loader.matrix(name, rows, rank)
}

fn rwkv_lora_rank(width: usize, scale: f32) -> usize {
    if width < 128 {
        8
    } else {
        let raw = scale * (width as f32).sqrt();
        (((raw / 32.0).round() as usize).max(1)) * 32
    }
}

#[cfg(test)]
mod runtime_state_tests {
    use super::*;

    #[test]
    fn explicit_output_packed_state_roundtrips_pytorch_layout_exactly() {
        let width = 8;
        let head_size = 4;
        let mut state = RwkvState::zeros(width, head_size);
        for index in 0..width {
            state.prev_tm[index] = index as f32 + 0.1;
            state.prev_cm[index] = index as f32 + 0.2;
            state.v_first[index] = index as f32 + 0.3;
            state.output[index] = index as f32 + 0.4;
        }
        for (index, value) in state.matrix.iter_mut().enumerate() {
            *value = index as f32 + 10.0;
        }

        let packed = state
            .to_explicit_output_packed(width, head_size)
            .expect("packing coherent-v9 state");
        assert_eq!(packed.len(), width * (4 + head_size));
        assert_eq!(&packed[0..8], &[0.1, 0.2, 0.3, 0.4, 10.0, 11.0, 12.0, 13.0]);
        assert_eq!(
            &packed[8..16],
            &[1.1, 1.2, 1.3, 1.4, 14.0, 15.0, 16.0, 17.0]
        );

        let restored = RwkvState::from_explicit_output_packed(width, head_size, &packed)
            .expect("restoring coherent-v9 state");
        assert_eq!(restored.prev_tm, state.prev_tm);
        assert_eq!(restored.prev_cm, state.prev_cm);
        assert_eq!(restored.v_first, state.v_first);
        assert_eq!(restored.output, state.output);
        assert_eq!(restored.matrix, state.matrix);
    }

    #[test]
    fn packed_state_rejects_wrong_shape_and_nonfinite_values() {
        assert!(RwkvState::from_explicit_output_packed(8, 4, &[0.0; 7]).is_err());
        let mut packed = vec![0.0; 8 * 8];
        packed[9] = f32::NAN;
        assert!(RwkvState::from_explicit_output_packed(8, 4, &packed).is_err());
    }
}
