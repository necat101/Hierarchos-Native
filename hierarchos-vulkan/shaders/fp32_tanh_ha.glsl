// High-accuracy binary32 tanh using a two-float exponential.
// Exp's degree-14 Taylor series is evaluated on |r| <= ln(2)/2.
// Coefficients are the nearest two-float expansions of the exact 1/n!.
// This path targets the high-accuracy MKL VML tanh used by FP32 PyTorch unary
// dispatch. Reuse the two-float helpers from fp32_sleef_tanh.glsl, but retain
// more exponential terms than the u10 approximation. No FP64 is required.
// The including RDNA3-only modules supply native IEEE FP32 FMA.
float cpu_tanh_ha(float x) {
    uint bits=floatBitsToUint(x), magnitude=bits&0x7fffffffu;
    if (magnitude>0x7f800000u) return x;
    if (magnitude<=0x39800000u) return x; // |x| <= 2^-12 rounds tanh(x) to x.
    if (magnitude>=0x41200000u) return uintBitsToFloat((bits&0x80000000u)|0x3f800000u);
    float ax=uintBitsToFloat(magnitude);
    precise float qf=roundEven(ax*1.44269504088896340736f);
    vec2 r=sleef_tanh_df_add2(vec2(ax,0.0f),-qf*0.693147182464599609375f);
    r=sleef_tanh_df_add2(r,qf*1.90465421212593355449e-9f);
    vec2 p=vec2(1.14707453605089604309e-11f, 2.37220768923123805476e-19f);
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(1.60590443720742825917e-10f, -5.35252651156272562627e-18f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(2.08767558795841523533e-09f, 1.10828391474598523083e-16f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(2.50521079436794025241e-08f, 4.41762304464836646350e-16f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(2.75573199814971303567e-07f, -7.57511220905119492253e-15f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(2.75573188446287531406e-06f, 3.79357122429722914969e-14f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(2.48015876422869041562e-05f, -3.40699609366681976930e-13f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(1.98412701138295233250e-04f, -2.72559687493345581544e-12f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(1.38888892251998186111e-03f, -3.36310944371032150002e-11f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(8.33333376795053482056e-03f, -4.34617203337595015000e-10f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(4.16666679084300994873e-02f, -1.24176346982807217501e-09f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(1.66666671633720397949e-01f, -4.96705387931228870002e-09f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(5.00000000000000000000e-01f, 0.00000000000000000000e+00f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(1.00000000000000000000e+00f, 0.00000000000000000000e+00f));
    p=sleef_tanh_df_add2(sleef_tanh_df_mul(p,r),vec2(1.00000000000000000000e+00f, 0.00000000000000000000e+00f));
    precise float scale=sleef_tanh_pow2i(int(qf));
    vec2 d=vec2(p.x*scale,p.y*scale);
    vec2 e=sleef_tanh_df_recip(d);
    vec2 numerator=sleef_tanh_df_add(d,vec2(-e.x,-e.y));
    vec2 denominator=sleef_tanh_df_add(d,e);
    vec2 ratio=sleef_tanh_df_div(numerator,denominator);
    precise float y=ratio.x+ratio.y;
    return uintBitsToFloat((floatBitsToUint(y)&0x7fffffffu)|(bits&0x80000000u));
}
