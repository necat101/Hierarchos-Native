// Gen9's native fma does not retain a full FP32 product. Preserve the CPU
// FMA sequence for full-precision operands, while keeping BF16/packed products
// on the native path: their product fits exactly in the FP32 significand.
#include "fp32_fma_exact.glsl"
float v41_gen9_fma(float a, float b, float c) {
    if (((floatBitsToUint(a) | floatBitsToUint(b)) & 0xffffu) == 0u)
        return fma(a, b, c);
    return fp32_fma_exact(a, b, c);
}
