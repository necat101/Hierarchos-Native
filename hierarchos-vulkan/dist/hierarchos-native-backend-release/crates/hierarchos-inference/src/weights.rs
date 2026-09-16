use std::{fs::File, path::Path, sync::Arc};

use memmap2::{Mmap, MmapOptions};
use safetensors::{tensor::Dtype, SafeTensors};

use crate::{
    error::{Error, Result},
    math::{Linear, Matrix},
};

pub(crate) struct WeightLoader {
    mmap: Mmap,
}

impl WeightLoader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only and the File is not mutated by this runtime.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        SafeTensors::deserialize(&mmap)?;
        Ok(Self { mmap })
    }

    fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Arc<[f32]>)> {
        let tensors = SafeTensors::deserialize(&self.mmap)?;
        let view = tensors
            .tensor(name)
            .map_err(|_| Error::MissingTensor(name.to_string()))?;
        let shape = view.shape().to_vec();
        let bytes = view.data();
        let mut values = Vec::with_capacity(match view.dtype() {
            Dtype::F32 => bytes.len() / 4,
            Dtype::F16 | Dtype::BF16 => bytes.len() / 2,
            dtype => {
                return Err(Error::Dtype {
                    name: name.to_string(),
                    dtype: format!("{dtype:?}"),
                })
            }
        });
        match view.dtype() {
            Dtype::F32 => {
                if bytes.len() % 4 != 0 {
                    return Err(Error::Invalid(format!(
                        "tensor '{name}' has invalid FP32 byte length {}",
                        bytes.len()
                    )));
                }
                for chunk in bytes.chunks_exact(4) {
                    values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            Dtype::F16 => {
                if bytes.len() % 2 != 0 {
                    return Err(Error::Invalid(format!(
                        "tensor '{name}' has invalid FP16 byte length {}",
                        bytes.len()
                    )));
                }
                for chunk in bytes.chunks_exact(2) {
                    values.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
                }
            }
            Dtype::BF16 => {
                if bytes.len() % 2 != 0 {
                    return Err(Error::Invalid(format!(
                        "tensor '{name}' has invalid BF16 byte length {}",
                        bytes.len()
                    )));
                }
                for chunk in bytes.chunks_exact(2) {
                    values.push(f32::from_bits(
                        u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16,
                    ));
                }
            }
            _ => unreachable!("dtype validated above"),
        }
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::Invalid(format!(
                    "tensor '{name}' contains a non-finite {:?} value at element {index}",
                    view.dtype()
                )));
            }
        }
        Ok((shape, Arc::from(values)))
    }

    pub fn vector(&self, name: &str, len: usize) -> Result<Arc<[f32]>> {
        let (shape, values) = self.tensor_f32(name)?;
        let shape_ok = shape == [len] || (len == 1 && shape.is_empty()) || shape == [1, len];
        if !shape_ok {
            return Err(Error::Shape {
                name: name.to_string(),
                actual: shape,
                expected: vec![len],
            });
        }
        Ok(values)
    }

    pub fn scalar(&self, name: &str) -> Result<f32> {
        Ok(self.vector(name, 1)?[0])
    }

    pub fn flat(&self, name: &str, len: usize) -> Result<Arc<[f32]>> {
        let (shape, values) = self.tensor_f32(name)?;
        if values.len() != len {
            return Err(Error::Shape {
                name: name.to_string(),
                actual: shape,
                expected: vec![len],
            });
        }
        Ok(values)
    }

    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        let (shape, values) = self.tensor_f32(name)?;
        if shape != [rows, cols] {
            return Err(Error::Shape {
                name: name.to_string(),
                actual: shape,
                expected: vec![rows, cols],
            });
        }
        Matrix::new(rows, cols, values)
    }

    pub fn linear(&self, prefix: &str, out: usize, input: usize, has_bias: bool) -> Result<Linear> {
        let weight = self.matrix(&format!("{prefix}.weight"), out, input)?;
        let bias = if has_bias {
            Some(self.vector(&format!("{prefix}.bias"), out)?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let encoded = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut mantissa = u32::from(fraction);
            let mut exponent32 = 113u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent32 -= 1;
            }
            mantissa &= 0x03ff;
            sign | (exponent32 << 23) | (mantissa << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (u32::from(fraction) << 13),
    };
    f32::from_bits(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decoder_handles_normal_and_subnormal_values() {
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
        assert_eq!(f16_bits_to_f32(0x0001), 2f32.powi(-24));
    }
}
