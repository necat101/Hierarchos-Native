// Non-last-axis CPU softmax vectorizes over channels, one host SIMD vector
// at a time. Remaining channels use std::exp's scalar FP32 rounding. This is
// distinct from TensorIterator's two-vector sigmoid/SiLU body.
#include "fp32_v41_unary.glsl"
float v41_compressor_exp(float x,uint channel,uint dim,uint vector_width) {
    uint width=max(vector_width,1u);
    return channel<(dim/width)*width ? sleef_expf_u10(x) : v41_scalar_exp(x);
}
