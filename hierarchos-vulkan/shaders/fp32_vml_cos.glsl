// Bounded cosine contract of the pinned MKL VML 2026.1 HA FP32 dispatcher.
// Its quadrant is FP32; its split-pi reduction and degree-nine polynomial
// use FP64 before the FP32 result. Constants are library coefficients, not
// fitted to model values. Independent CPU samples and RoPE phases qualify it.
// The including shader supplies fp32_cos for the unmeasured large-angle path.
#include "fp32_fma_exact.glsl"
float vml_cos_bounded(float input_value) {
    if (!(abs(input_value) <= 8.0)) return fp32_cos(input_value);
    float value = abs(input_value);
    precise float shifted = value + uintBitsToFloat(0x3fc90fdbu);
    float magic = fp32_fma_exact(shifted, uintBitsToFloat(0x3ea2f983u), -12582912.0);
    precise float quadrant = magic + 12582912.0;
    precise float n = quadrant - 0.5;
    precise double r = fma(-double(n), packDouble2x32(uvec2(0x54440000u, 0x400921fbu)), double(value));
    r = fma(-double(n), packDouble2x32(uvec2(0x4c4c6629u, 0x3d968c23u)), r);
    precise double square = r * r;
    precise double polynomial = packDouble2x32(uvec2(0x0e4c7deeu, 0x3ec5dbdfu));
    polynomial = fma(polynomial, square, packDouble2x32(uvec2(0xeea73463u, 0xbf29f6ffu)));
    polynomial = fma(polynomial, square, packDouble2x32(uvec2(0x3804ca96u, 0x3f8110edu)));
    polynomial = fma(polynomial, square, packDouble2x32(uvec2(0xbc836587u, 0xbfc55554u)));
    polynomial = polynomial * square;
    precise double result = fma(polynomial, r, r);
    return uintBitsToFloat(floatBitsToUint(float(result)) ^ (floatBitsToUint(magic) << 31u));
}
