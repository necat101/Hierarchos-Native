// Scalar GLSL adaptation of SLEEF 3.6.1 xtanhf (u10), matching the
// AVX512 float tanh path used by the PyTorch CPU oracle.  SLEEF revision:
// 60e76d2bce17d278b439d9da17177c8f957a9e9b.
//
// The compensated two-float operations are intentional.  A plain GLSL
// tanh() differs by a few FP32 ulps for Gemma-style GELU-tanh inputs, and the
// following RMSNorm can amplify that otherwise tiny activation drift.
//
// Two staging rules are load-bearing here:
//   * every literal carries an explicit FP32 suffix.  A bare double literal
//     promotes the surrounding expression to FP64; a Gen9 iGPU emulates that
//     (and a double-capable device runs it natively), which lands on the
//     correctly rounded value instead of SLEEF's staged FP32 one.
//   * every multiply-add is fp32_fma_exact.  The df_* helpers recover the
//     exact product error through fma(a, b, -hi); with a truncated MAD the
//     error term is wrong and the whole compensated chain drifts by an ulp.
//   * the two reciprocals are fp32_div.  A bare `/` lowers to a
//     reciprocal-multiply on this device, which is one ulp away from the
//     correctly rounded quotient SLEEF's vrec/vdiv produce on the CPU.
//
// fp32_fma_exact and fp32_div are supplied by fp32_fma_exact.glsl and
// fp32_sqrt_recip.glsl, which the including shader must pull in first.

float sleef_tanh_pow2i(int q) {
    if (q > 127) return uintBitsToFloat(0x7f800000u);
    if (q >= -126) return uintBitsToFloat(uint(q + 127) << 23u);
    if (q < -149) return 0.0f;
    return uintBitsToFloat(1u << uint(q + 149));
}

vec2 sleef_tanh_df_add2(vec2 x, float y) {
    precise float hi = x.x + y;
    precise float v = hi - x.x;
    precise float lo = (x.x - (hi - v)) + (y - v);
    lo = lo + x.y;
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_add(vec2 x, vec2 y) {
    // SLEEF dfadd_vf2_vf2_vf2: |x| >= |y| for its call sites here.
    precise float hi = x.x + y.x;
    precise float lo = (x.x - hi) + y.x;
    lo = lo + x.y;
    lo = lo + y.y;
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_add2(vec2 x, vec2 y) {
    precise float hi = x.x + y.x;
    precise float v = hi - x.x;
    precise float lo = (x.x - (hi - v)) + (y.x - v);
    precise float tails = x.y + y.y;
    lo = lo + tails;
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_add(float x, vec2 y) {
    precise float hi = x + y.x;
    precise float lo = (x - hi) + y.x;
    lo = lo + y.y;
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_mul(vec2 x, float y) {
    precise float hi = x.x * y;
    precise float lo = fp32_fma_exact(x.x, y, -hi);
    lo = fp32_fma_exact(x.y, y, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_mul(vec2 x, vec2 y) {
    precise float hi = x.x * y.x;
    precise float lo = fp32_fma_exact(x.x, y.x, -hi);
    lo = fp32_fma_exact(x.y, y.x, lo);
    lo = fp32_fma_exact(x.x, y.y, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_square(vec2 x) {
    precise float hi = x.x * x.x;
    precise float lo = fp32_fma_exact(x.x, x.x, -hi);
    precise float twice = x.x + x.x;
    lo = fp32_fma_exact(twice, x.y, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_recip(vec2 d) {
    precise float hi = fp32_div(1.0f, d.x);
    precise float residual = fp32_fma_exact(-d.x, hi, 1.0f);
    residual = fp32_fma_exact(-d.y, hi, residual);
    precise float lo = hi * residual;
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_div(vec2 n, vec2 d) {
    precise float t = fp32_div(1.0f, d.x);
    precise float hi = n.x * t;
    precise float u = fp32_fma_exact(t, n.x, -hi);
    precise float v = fp32_fma_exact(-d.x, t, 1.0f);
    v = fp32_fma_exact(-d.y, t, v);
    precise float lo = fp32_fma_exact(n.y, t, u);
    lo = fp32_fma_exact(hi, v, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_expk2(vec2 d) {
    precise float summed = d.x + d.y;
    precise float qf = roundEven(summed * 1.4426950408889634073599246810018921f);
    int q = int(qf);

    vec2 s = sleef_tanh_df_add2(d, qf * -0.693145751953125f);
    s = sleef_tanh_df_add2(s, qf * -1.428606765330187045e-06f);

    precise float u = 0.1980960224e-3f;
    u = fp32_fma_exact(u, s.x, 0.1394256484e-2f);
    u = fp32_fma_exact(u, s.x, 0.8333456703e-2f);
    u = fp32_fma_exact(u, s.x, 0.4166637361e-1f);

    vec2 t = sleef_tanh_df_add2(sleef_tanh_df_mul(s, u), 0.166666659414234244790680580464f);
    t = sleef_tanh_df_add2(sleef_tanh_df_mul(s, t), 0.5f);
    t = sleef_tanh_df_add2(s, sleef_tanh_df_mul(sleef_tanh_df_square(s), t));
    t = sleef_tanh_df_add(1.0f, t);

    precise float scale = sleef_tanh_pow2i(q);
    t.x = t.x * scale;
    t.y = t.y * scale;
    if (d.x < -104.0f) return vec2(0.0f);
    return t;
}

float sleef_tanhf_u10(float x) {
    if (isnan(x)) return x;
    precise float ax = abs(x);
    vec2 d = sleef_tanh_expk2(vec2(ax, 0.0f));
    vec2 e = sleef_tanh_df_recip(d);
    vec2 neg_e = vec2(-e.x, -e.y);
    vec2 numerator = sleef_tanh_df_add(d, neg_e);
    vec2 denominator = sleef_tanh_df_add(d, e);
    vec2 ratio = sleef_tanh_df_div(numerator, denominator);
    precise float y = ratio.x + ratio.y;
    if (ax > 8.664339742f) y = 1.0f;
    uint sign_bit = floatBitsToUint(x) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(y) & 0x7fffffffu) | sign_bit);
}
