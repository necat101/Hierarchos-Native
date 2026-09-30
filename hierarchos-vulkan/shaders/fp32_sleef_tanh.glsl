// Scalar GLSL adaptation of SLEEF 3.6.1 xtanhf (u10), matching the
// AVX512 float tanh path used by the PyTorch CPU oracle.  SLEEF revision:
// 60e76d2bce17d278b439d9da17177c8f957a9e9b.
//
// The compensated two-float operations are intentional.  A plain GLSL
// tanh() differs by a few FP32 ulps for Gemma-style GELU-tanh inputs, and the
// following RMSNorm can amplify that otherwise tiny activation drift.

float sleef_tanh_pow2i(int q) {
    if (q > 127) return uintBitsToFloat(0x7f800000u);
    if (q >= -126) return uintBitsToFloat(uint(q + 127) << 23u);
    if (q < -149) return 0.0;
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
    precise float lo = fma(x.x, y, -hi);
    lo = fma(x.y, y, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_mul(vec2 x, vec2 y) {
    precise float hi = x.x * y.x;
    precise float lo = fma(x.x, y.x, -hi);
    lo = fma(x.y, y.x, lo);
    lo = fma(x.x, y.y, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_square(vec2 x) {
    precise float hi = x.x * x.x;
    precise float lo = fma(x.x, x.x, -hi);
    precise float twice = x.x + x.x;
    lo = fma(twice, x.y, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_recip(vec2 d) {
    precise float hi = 1.0 / d.x;
    precise float residual = fma(-d.x, hi, 1.0);
    residual = fma(-d.y, hi, residual);
    precise float lo = hi * residual;
    return vec2(hi, lo);
}

vec2 sleef_tanh_df_div(vec2 n, vec2 d) {
    precise float t = 1.0 / d.x;
    precise float hi = n.x * t;
    precise float u = fma(t, n.x, -hi);
    precise float v = fma(-d.x, t, 1.0);
    v = fma(-d.y, t, v);
    precise float lo = fma(n.y, t, u);
    lo = fma(hi, v, lo);
    return vec2(hi, lo);
}

vec2 sleef_tanh_expk2(vec2 d) {
    precise float summed = d.x + d.y;
    precise float qf = roundEven(summed * 1.4426950408889634073599246810018921);
    int q = int(qf);

    vec2 s = sleef_tanh_df_add2(d, qf * -0.693145751953125);
    s = sleef_tanh_df_add2(s, qf * -1.428606765330187045e-06);

    precise float u = 0.1980960224e-3;
    u = fma(u, s.x, 0.1394256484e-2);
    u = fma(u, s.x, 0.8333456703e-2);
    u = fma(u, s.x, 0.4166637361e-1);

    vec2 t = sleef_tanh_df_add2(sleef_tanh_df_mul(s, u), 0.166666659414234244790680580464);
    t = sleef_tanh_df_add2(sleef_tanh_df_mul(s, t), 0.5);
    t = sleef_tanh_df_add2(s, sleef_tanh_df_mul(sleef_tanh_df_square(s), t));
    t = sleef_tanh_df_add(1.0, t);

    precise float scale = sleef_tanh_pow2i(q);
    t.x = t.x * scale;
    t.y = t.y * scale;
    if (d.x < -104.0) return vec2(0.0);
    return t;
}

float sleef_tanhf_u10(float x) {
    if (isnan(x)) return x;
    precise float ax = abs(x);
    vec2 d = sleef_tanh_expk2(vec2(ax, 0.0));
    vec2 e = sleef_tanh_df_recip(d);
    vec2 neg_e = vec2(-e.x, -e.y);
    vec2 numerator = sleef_tanh_df_add(d, neg_e);
    vec2 denominator = sleef_tanh_df_add(d, e);
    vec2 ratio = sleef_tanh_df_div(numerator, denominator);
    precise float y = ratio.x + ratio.y;
    if (ax > 8.664339742) y = 1.0;
    uint sign_bit = floatBitsToUint(x) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(y) & 0x7fffffffu) | sign_bit);
}
