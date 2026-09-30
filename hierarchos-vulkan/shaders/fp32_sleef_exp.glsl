// FP32 exp path matching the SLEEF expf_u10 polynomial used by PyTorch's
// AVX512 Vectorized<float>::exp().  Coefficients and staging correspond to
// SLEEF revision 60e76d2bce17d278b439d9da17177c8f957a9e9b (PyTorch 2.6.0).
// Keep every intermediate in FP32 and use explicit fma where SLEEF does.

float sleef_pow2i(int q) {
    if (q > 127) {
        return uintBitsToFloat(0x7f800000u);
    }
    if (q >= -126) {
        return uintBitsToFloat(uint(q + 127) << 23u);
    }
    if (q < -149) {
        return 0.0;
    }
    return uintBitsToFloat(1u << uint(q + 149));
}

float sleef_expf_u10(float d) {
    // SLEEF's vrint is round-to-nearest, ties-to-even.
    int q = int(roundEven(d * 1.44269504088896340736));
    precise float s = fma(float(q), -0.693145751953125, d);
    s = fma(float(q), -1.428606765330187045e-06, s);

    precise float u = 0.000198527617612853646278381;
    u = fma(u, s, 0.00139304355252534151077271);
    u = fma(u, s, 0.00833336077630519866943359);
    u = fma(u, s, 0.0416664853692054748535156);
    u = fma(u, s, 0.166666671633720397949219);
    u = fma(u, s, 0.5);
    u = 1.0 + fma(s * s, u, s);
    u *= sleef_pow2i(q);

    if (d < -104.0) {
        return 0.0;
    }
    if (d > 100.0) {
        return uintBitsToFloat(0x7f800000u);
    }
    return u;
}

// Positive, finite, normal-domain portion of SLEEF xlogf for AVX512F. This
// is the only domain needed by log(sum(exp(...))) in Falcon H1 cross entropy.
float sleef_logf_u10_positive(float d) {
    // AVX512 SLEEF uses getexp(d * (1 / 0.75)) and getmant(d) normalized to
    // [0.75, 1.5). For a positive normal float, extracting the binary exponent
    // from d*(4/3) and scaling d by the inverse power of two is identical.
    precise float scaled = d * 1.33333333333333333333;
    uint scaled_bits = floatBitsToUint(scaled);
    int e = int((scaled_bits >> 23u) & 0xffu) - 127;
    precise float m = d * sleef_pow2i(-e);

    precise float x = fp32_div(m - 1.0, 1.0 + m);
    precise float x2 = x * x;
    precise float t = 0.2392828464508056640625;
    t = fma(t, x2, 0.28518211841583251953125);
    t = fma(t, x2, 0.400005877017974853515625);
    t = fma(t, x2, 0.666666686534881591796875);
    t = fma(t, x2, 2.0);
    precise float e_ln2 = float(e) * 0.693147180559945286226764;
    return fma(x, t, e_ln2);
}
