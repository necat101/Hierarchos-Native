// Scalar GLSL adaptation of the positive-normal SLEEF 3.6 xpowf (u10) path
// used by PyTorch 2.6 AVX512 Vectorized<float>::pow().
//
// PEFT Gemma4 only needs x > 0 and y == -1.5 here (the derivative of
// torch.pow(x, -0.5)).  Keeping this helper domain-specific avoids importing
// SLEEF's unrelated NaN/sign/integer-exponent cases while preserving the
// compensated log -> multiply -> exp arithmetic that affects FP32 parity.

float sleef_pow_pow2i(int q) {
    if (q > 127) return uintBitsToFloat(0x7f800000u);
    if (q >= -126) return uintBitsToFloat(uint(q + 127) << 23u);
    if (q < -149) return 0.0;
    return uintBitsToFloat(1u << uint(q + 149));
}

vec2 sleef_pow_df_normalize(vec2 x) {
    precise float hi = x.x + x.y;
    precise float lo = (x.x - hi) + x.y;
    return vec2(hi, lo);
}

vec2 sleef_pow_df_scale(vec2 x, float y) {
    return vec2(x.x * y, x.y * y);
}

vec2 sleef_pow_df_add2(float x, float y) {
    precise float hi = x + y;
    precise float v = hi - x;
    precise float lo = (x - (hi - v)) + (y - v);
    return vec2(hi, lo);
}

vec2 sleef_pow_df_add2(vec2 x, float y) {
    precise float hi = x.x + y;
    precise float v = hi - x.x;
    precise float lo = (x.x - (hi - v)) + (y - v);
    lo = lo + x.y;
    return vec2(hi, lo);
}

vec2 sleef_pow_df_add2(vec2 x, vec2 y) {
    precise float hi = x.x + y.x;
    precise float v = hi - x.x;
    precise float lo = (x.x - (hi - v)) + (y.x - v);
    lo = lo + (x.y + y.y);
    return vec2(hi, lo);
}

vec2 sleef_pow_df_add(vec2 x, vec2 y) {
    // SLEEF dfadd_vf2_vf2_vf2. Its pow call sites satisfy |x| >= |y|.
    precise float hi = x.x + y.x;
    precise float lo = (x.x - hi) + y.x;
    lo = lo + x.y;
    lo = lo + y.y;
    return vec2(hi, lo);
}

vec2 sleef_pow_df_add(float x, vec2 y) {
    precise float hi = x + y.x;
    precise float lo = (x - hi) + y.x;
    lo = lo + y.y;
    return vec2(hi, lo);
}

vec2 sleef_pow_df_mul(vec2 x, float y) {
    precise float hi = x.x * y;
    precise float lo = fma(x.x, y, -hi);
    lo = fma(x.y, y, lo);
    return vec2(hi, lo);
}

vec2 sleef_pow_df_mul(vec2 x, vec2 y) {
    precise float hi = x.x * y.x;
    precise float lo = fma(x.x, y.x, -hi);
    lo = fma(x.y, y.x, lo);
    lo = fma(x.x, y.y, lo);
    return vec2(hi, lo);
}

vec2 sleef_pow_df_square(vec2 x) {
    precise float hi = x.x * x.x;
    precise float lo = fma(x.x, x.x, -hi);
    lo = fma(x.x + x.x, x.y, lo);
    return vec2(hi, lo);
}

vec2 sleef_pow_df_div(vec2 n, vec2 d) {
    precise float t = 1.0 / d.x;
    precise float hi = n.x * t;
    precise float u = fma(t, n.x, -hi);
    precise float v = fma(-d.x, t, 1.0);
    v = fma(-d.y, t, v);
    precise float lo = fma(n.y, t, u);
    lo = fma(hi, v, lo);
    return vec2(hi, lo);
}

vec2 sleef_pow_logkf_positive(float d) {
    // AVX512 logkf uses getexp(d * 4/3) and getmant(d, p75_1p5).
    // For positive normal FP32 this bit extraction/scaling is equivalent.
    precise float scaled = d * 1.33333333333333333333;
    uint scaled_bits = floatBitsToUint(scaled);
    int e = int((scaled_bits >> 23u) & 0xffu) - 127;
    precise float m = d * sleef_pow_pow2i(-e);

    vec2 x = sleef_pow_df_div(
        sleef_pow_df_add2(-1.0, m),
        sleef_pow_df_add2(1.0, m)
    );
    vec2 x2 = sleef_pow_df_square(x);

    precise float t = 0.240320354700088500976562;
    t = fma(t, x2.x, 0.285112679004669189453125);
    t = fma(t, x2.x, 0.400007992982864379882812);
    vec2 c = vec2(
        0.66666662693023681640625,
        3.69183861259614332084311e-09
    );

    vec2 s = sleef_pow_df_mul(
        vec2(0.69314718246459960938, -1.904654323148236017e-09),
        float(e)
    );
    s = sleef_pow_df_add(s, sleef_pow_df_scale(x, 2.0));
    vec2 correction = sleef_pow_df_mul(
        sleef_pow_df_mul(x2, x),
        sleef_pow_df_add2(sleef_pow_df_mul(x2, t), c)
    );
    return sleef_pow_df_add(s, correction);
}

float sleef_pow_expkf(vec2 d) {
    precise float summed = d.x + d.y;
    int q = int(roundEven(summed * 1.44269504088896340736));
    precise float qf = float(q);

    vec2 s = sleef_pow_df_add2(d, qf * -0.693145751953125);
    s = sleef_pow_df_add2(s, qf * -1.428606765330187045e-06);
    s = sleef_pow_df_normalize(s);

    precise float u = 0.00136324646882712841033936;
    u = fma(u, s.x, 0.00836596917361021041870117);
    u = fma(u, s.x, 0.0416710823774337768554688);
    u = fma(u, s.x, 0.166665524244308471679688);
    u = fma(u, s.x, 0.499999850988388061523438);

    vec2 t = sleef_pow_df_add(
        s,
        sleef_pow_df_mul(sleef_pow_df_square(s), u)
    );
    t = sleef_pow_df_add(1.0, t);
    precise float result = (t.x + t.y) * sleef_pow_pow2i(q);
    if (d.x < -104.0) return 0.0;
    return result;
}

float sleef_powf_positive_u10(float x, float y) {
    vec2 log_x = sleef_pow_logkf_positive(x);
    vec2 exponent = sleef_pow_df_mul(log_x, y);
    return sleef_pow_expkf(exponent);
}

float sleef_powf_neg_one_half_u10(float x) {
    return sleef_powf_positive_u10(x, -0.5);
}

float sleef_powf_neg_three_halves_u10(float x) {
    return sleef_powf_positive_u10(x, -1.5);
}
