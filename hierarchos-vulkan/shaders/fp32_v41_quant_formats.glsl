// Shared released E4M3/E2M1 and UE8M0 conversion rules.
uint round_even_nonnegative(float v) {
    float lower = floor(v);
    uint base = uint(lower);
    float fraction = v - lower;
    if (fraction > 0.5) return base + 1u;
    if (fraction == 0.5 && (base & 1u) != 0u) return base + 1u;
    return base;
}

// E4M3 finite (fn) encoding with round-to-nearest-even. Values are clamped by
// the caller, so the 0x7f NaN pattern is never produced.
uint encode_e4m3(float v) {
    uint sign_bit = (v < 0.0) ? 0x80u : 0u;
    float a = abs(v);
    if (a > 448.0) a = 448.0;
    if (a < 0.015625) {  // below the smallest normal (2^-6)
        // Subnormal grid step is 2^-9; m == 8 promotes to the normal 0x08 code.
        uint m = round_even_nonnegative(a * 512.0);
        return sign_bit | min(m, 8u);
    }
    int exponent;
    float significand = frexp(a, exponent);  // a = significand * 2^exponent
    int unbiased = exponent - 1;             // normalized significand in [1, 2)
    uint mantissa = round_even_nonnegative((significand * 2.0 - 1.0) * 8.0);
    if (mantissa == 8u) {
        mantissa = 0u;
        unbiased += 1;
    }
    // a is at most 448, so the biased exponent stays in [1, 15]; 15/7 would be
    // the reserved NaN pattern and cannot be reached by a clamped finite input.
    return sign_bit | (uint(unbiased + 7) << 3) | mantissa;
}

float decode_e4m3(uint bits) {
    float sign = ((bits & 0x80u) != 0u) ? -1.0 : 1.0;
    uint exponent = (bits >> 3) & 0xFu;
    uint mantissa = bits & 0x7u;
    if (exponent == 0u) return sign * float(mantissa) * (1.0 / 512.0);
    return sign * ldexp(1.0 + float(mantissa) * 0.125, int(exponent) - 7);
}

// E2M1 codes: 0, .5, 1, 1.5, 2, 3, 4, 6; ties round to the even code.
uint encode_e2m1(float v) {
    uint sign_bit = (v < 0.0) ? 0x8u : 0u;
    float a = abs(v);
    if (a > 6.0) a = 6.0;
    uint code;
    if (a < 0.75) {
        code = (a <= 0.25) ? 0u : 1u;
    } else if (a < 1.75) {
        code = (a < 1.25) ? 2u : 3u;
    } else if (a < 3.5) {
        code = (a < 2.5) ? 4u : 5u;
    } else if (a < 5.0) {
        code = 6u;
    } else {
        code = 7u;
    }
    return sign_bit | code;
}

float decode_e2m1(uint bits) {
    float sign = ((bits & 0x8u) != 0u) ? -1.0 : 1.0;
    uint code = bits & 0x7u;
    float value;
    if (code == 0u) value = 0.0;
    else if (code == 1u) value = 0.5;
    else if (code == 2u) value = 1.0;
    else if (code == 3u) value = 1.5;
    else if (code == 4u) value = 2.0;
    else if (code == 5u) value = 3.0;
    else if (code == 6u) value = 4.0;
    else value = 6.0;
    return sign * value;
}

// fast_log2_ceil / fast_pow2 from the released kernel: ceil(log2(v)) by exponent
// extraction, then 2^e by exponent insertion.
float round_scale_pow2(float v) {
    uint bits = floatBitsToUint(v);
    int exponent = int((bits >> 23) & 0xFFu) - 127;
    if ((bits & 0x7FFFFFu) != 0u) exponent += 1;
    return uintBitsToFloat(uint(exponent + 127) << 23);
}

