use rayon::prelude::*;
use std::sync::Arc;

use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Arc<[f32]>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Arc<[f32]>) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(Error::Invalid(format!(
                "matrix {}x{} has {} values",
                rows,
                cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    #[inline]
    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Conventional matrix-vector multiply for PyTorch Linear weights [out, in].
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        if self.rows >= 2_048 {
            (0..self.rows)
                .into_par_iter()
                .map(|row| dot(self.row(row), x))
                .collect()
        } else {
            (0..self.rows).map(|row| dot(self.row(row), x)).collect()
        }
    }

    /// Row-vector times a row-major parameter matrix [in, out].
    pub fn row_vec_mat(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.rows);
        let mut out = vec![0.0f32; self.cols];
        for (row_idx, &scale) in x.iter().enumerate() {
            let row = self.row(row_idx);
            for col in 0..self.cols {
                out[col] = scale.mul_add(row[col], out[col]);
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Linear {
    pub weight: Matrix,
    pub bias: Option<Arc<[f32]>>,
}

impl Linear {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.weight.matvec(x);
        if let Some(bias) = &self.bias {
            debug_assert_eq!(out.len(), bias.len());
            for (value, &b) in out.iter_mut().zip(bias.iter()) {
                *value += b;
            }
        }
        out
    }
}

#[inline(always)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0usize;
    while i + 4 <= a.len() {
        s0 = a[i].mul_add(b[i], s0);
        s1 = a[i + 1].mul_add(b[i + 1], s1);
        s2 = a[i + 2].mul_add(b[i + 2], s2);
        s3 = a[i + 3].mul_add(b[i + 3], s3);
        i += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while i < a.len() {
        sum = a[i].mul_add(b[i], sum);
        i += 1;
    }
    sum
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub(crate) fn layer_norm(
    x: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    let inv_n = 1.0 / x.len() as f32;
    let mean = x.iter().copied().sum::<f32>() * inv_n;
    let variance = x
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        * inv_n;
    let inv_std = 1.0 / (variance + eps).sqrt();
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let mut v = (x[i] - mean) * inv_std;
        if let Some(w) = weight {
            v *= w[i];
        }
        if let Some(b) = bias {
            v += b[i];
        }
        out.push(v);
    }
    out
}

pub(crate) fn group_norm(
    x: &[f32],
    groups: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(x.len() % groups, 0);
    let width = x.len() / groups;
    let mut out = vec![0.0f32; x.len()];
    for group in 0..groups {
        let start = group * width;
        let end = start + width;
        let slice = &x[start..end];
        let inv_n = 1.0 / width as f32;
        let mean = slice.iter().copied().sum::<f32>() * inv_n;
        let variance = slice
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            * inv_n;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for i in start..end {
            out[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
        }
    }
    out
}

#[inline]
pub(crate) fn finite_clamp(value: f32, max_abs: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-max_abs, max_abs)
    } else {
        value
    }
}

pub(crate) fn finite_clamp_vec(values: &mut [f32], max_abs: f32) {
    for value in values {
        *value = finite_clamp(*value, max_abs);
    }
}

pub(crate) fn l2_norm_clamp(values: &mut [f32], max_norm: f32) {
    if max_norm <= 0.0 {
        return;
    }
    let norm = values
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt() as f32;
    let scale = (max_norm / (norm + 1e-6)).min(1.0);
    if scale < 1.0 {
        for v in values {
            *v *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_has_zero_mean_and_unit_variance() {
        let out = layer_norm(&[1.0, 2.0, 3.0, 4.0], None, None, 1e-5);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-6);
        assert!((var - 1.0).abs() < 2e-5);
    }

    #[test]
    fn row_vec_mat_uses_parameter_layout() {
        let m = Matrix::new(2, 3, Arc::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        assert_eq!(m.row_vec_mat(&[2.0, 3.0]), vec![14.0, 19.0, 24.0]);
    }
}
