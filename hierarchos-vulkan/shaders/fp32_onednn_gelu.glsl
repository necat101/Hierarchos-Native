// Arithmetic staging of oneDNN 3.12's AVX2 GELU(erf) injector, revision
// 80afa71049cd69a3df32adcccb623b12cd7baa22. Coefficients are its FP32 words.
// Reference: src/cpu/x64/injectors/jit_uni_eltwise_injector.cpp (Apache-2.0).
#ifndef HIERARCHOS_FP32_ONEDNN_GELU
#define HIERARCHOS_FP32_ONEDNN_GELU
#include "fp32_fma_exact.glsl"
#include "fp32_sqrt_recip.glsl"

float onednn_exp_negative(float input_value) {
    bool underflow = input_value < uintBitsToFloat(0xc2aeac50u);
    float value = max(input_value, uintBitsToFloat(0xc2aeac50u));
    precise float scaled = value * uintBitsToFloat(0x3fb8aa3bu);
    precise float shifted = scaled + 0.5;
    float exponent = floor(shifted);
    float remainder = fp32_fma_exact(-exponent, uintBitsToFloat(0x3f317218u), value);
    float power = underflow ? 0.0 : uintBitsToFloat(uint(int(exponent) + 126) << 23u);
    float polynomial = uintBitsToFloat(0x3c07cfceu);
    polynomial = fp32_fma_exact(polynomial, remainder, uintBitsToFloat(0x3d2b9d0du));
    polynomial = fp32_fma_exact(polynomial, remainder, uintBitsToFloat(0x3e2aad40u));
    polynomial = fp32_fma_exact(polynomial, remainder, uintBitsToFloat(0x3efffee3u));
    polynomial = fp32_fma_exact(polynomial, remainder, uintBitsToFloat(0x3f7ffffbu));
    polynomial = fp32_fma_exact(polynomial, remainder, 1.0);
    precise float product = polynomial * power;
    precise float result = product * 2.0;
    return result;
}

float onednn_erf(float scaled_input, float exponential) {
    float denominator = fp32_fma_exact(uintBitsToFloat(0x3ea7ba05u), abs(scaled_input), 1.0);
    float inverse = fp32_div(1.0, denominator);
    float polynomial = uintBitsToFloat(0x3f87dc22u);
    polynomial = fp32_fma_exact(polynomial, inverse, uintBitsToFloat(0xbfba00e3u));
    polynomial = fp32_fma_exact(polynomial, inverse, uintBitsToFloat(0x3fb5f0e3u));
    polynomial = fp32_fma_exact(polynomial, inverse, uintBitsToFloat(0xbe91a98eu));
    polynomial = fp32_fma_exact(polynomial, inverse, uintBitsToFloat(0x3e827906u));
    precise float tail = (-exponential) * inverse;
    float result = fp32_fma_exact(tail, polynomial, 1.0);
    return uintBitsToFloat(floatBitsToUint(result) ^ (floatBitsToUint(scaled_input) & 0x80000000u));
}

float onednn_gelu_forward(float value) {
    if (abs(value) >= 20.0) return value > 0.0 ? value : 0.0;
    precise float scaled = value * uintBitsToFloat(0x3f3504f3u);
    precise float square = scaled * scaled;
    float exponential = onednn_exp_negative(-square);
    float error_function = onednn_erf(scaled, exponential);
    precise float half_value = value * 0.5;
    return fp32_fma_exact(error_function, half_value, half_value);
}

float onednn_gelu_backward(float value, float gradient) {
    if (abs(value) >= 20.0) return value > 0.0 ? gradient : gradient * 0.0;
    precise float scaled = value * uintBitsToFloat(0x3f3504f3u);
    precise float square = scaled * scaled;
    float exponential = onednn_exp_negative(-square);
    float error_function = onednn_erf(scaled, exponential);
    precise float density = scaled * uintBitsToFloat(0x3f106ebau);
    precise float term = density * exponential;
    precise float offset = term + 0.5;
    float derivative = fp32_fma_exact(error_function, 0.5, offset);
    precise float result = derivative * gradient;
    return result;
}
#endif
