// FP32 two-float scalar exp for the pinned CPU oracle's scalar sigmoid tail.
// Range-reduce by a two-float ln(2), then evaluate exp(r) to degree 14
// on |r| <= ln(2)/2. Every product residual uses the shared exact-FMA helper;
// no shaderFloat64 feature or host activation math is needed.
// The vector body retains the existing SLEEF expf_u10 staging. PyTorch's
// cpu_kernel_vec processes pairs of 16-wide AVX512 vectors, then scalar tails.
// Include fp32_sqrt_recip, fp32_fma_exact, and fp32_sleef_exp before this file.

vec2 v41_df_add(vec2 a, vec2 b) {
    precise float hi = a.x + b.x;
    precise float v = hi - a.x;
    precise float error = (a.x - (hi - v)) + (b.x - v);
    error = (error + a.y) + b.y;
    precise float result = hi + error;
    precise float tail = error - (result - hi);
    return vec2(result, tail);
}

vec2 v41_df_mul(vec2 a, vec2 b) {
    precise float hi = a.x * b.x;
    precise float error = fp32_fma_exact(a.x, b.x, -hi);
    error = error + a.x * b.y;
    error = error + a.y * b.x;
    error = error + a.y * b.y;
    precise float result = hi + error;
    precise float tail = error - (result - hi);
    return vec2(result, tail);
}

float v41_scalar_exp(float x) {
    if (isnan(x)) return x;
    if (x > 88.72283172607421875f) return uintBitsToFloat(0x7f800000u);
    if (x < -103.97208404541015625f) return 0.0f;
    int q = int(roundEven(x * 1.44269504088896340736f));
    vec2 reduced = v41_df_add(vec2(x, 0.0f),
        v41_df_mul(vec2(float(q), 0.0f),
                   vec2(-0.693147182464599609375f, 1.904654323148236e-9f)));
    vec2 p = vec2(1.147074536050896e-11f, 2.372207689231238e-19f);
    p = v41_df_add(v41_df_mul(p, reduced), vec2(1.6059044372074283e-10f, -5.352526511562726e-18f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(2.0876755879584152e-9f, 1.1082839809204342e-16f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(2.5052107943679403e-8f, 4.4176230446483665e-16f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(2.755731998149713e-7f, -7.575112209051195e-15f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.0000027557318844628753f, 3.793571224297229e-14f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.000024801587642286904f, -3.40699609366682e-13f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.00019841270113829523f, -2.725596874933456e-12f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.0013888889225199819f, -3.3631094437103215e-11f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.008333333767950535f, -4.34617203337595e-10f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.0416666679084301f, -1.2417634698280722e-9f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.1666666716337204f, -4.967053879312289e-9f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(0.5f, 0.0f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(1.0f, 0.0f));
    p = v41_df_add(v41_df_mul(p, reduced), vec2(1.0f, 0.0f));
    precise float rounded = p.x + p.y;
    return ldexp(rounded, q);
}

float v41_cpu_sigmoid(float x, uint index, uint elements) {
    uint vector_end = elements / 32u * 32u;
    float e = index < vector_end ? sleef_expf_u10(-x) : v41_scalar_exp(-x);
    return fp32_div(1.0f, 1.0f + e);
}

float v41_cpu_silu(float x, uint index, uint elements) {
    uint vector_end = elements / 32u * 32u;
    float e = index < vector_end ? sleef_expf_u10(-x) : v41_scalar_exp(-x);
    // PyTorch divides x by the denominator directly. x * sigmoid(x) adds
    // a separately rounded reciprocal and gives different FP32 results.
    return fp32_div(x, 1.0f + e);
}

vec2 v41_df_div(vec2 a, vec2 b) {
    precise float q = fp32_div(a.x, b.x);
    vec2 residual = v41_df_add(a, -v41_df_mul(vec2(q, 0.0f), b));
    precise float tail = fp32_div(residual.x + residual.y, b.x);
    return v41_df_add(vec2(q, 0.0f), vec2(tail, 0.0f));
}

// Independent two-float atanh series for scalar log1p on positive inputs.
float v41_df_log_positive(vec2 y) {
    int e = int(floatBitsToUint(y.x) >> 23u) - 127;
    y = vec2(ldexp(y.x, -e), ldexp(y.y, -e));
    vec2 u = v41_df_div(v41_df_add(y, vec2(-1.0f, 0.0f)),
                        v41_df_add(y, vec2(1.0f, 0.0f)));
    vec2 square = v41_df_mul(u, u);
    vec2 p = vec2(0.02857142873108387f, -1.5965530286710106e-10f);
    p = v41_df_add(v41_df_mul(p, square), vec2(0.03030303120613098f, -9.03100705329507e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.032258063554763794f, 9.613652096618353e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.03448275849223137f, 1.284582856753147e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.03703703731298447f, -2.759474315716659e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.03999999910593033f, 8.940696516468449e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.043478261679410934f, -8.098457460192776e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.0476190485060215f, -8.869738832295582e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.05263157933950424f, -3.9213582381236733e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.05882352963089943f, -2.1913472425527658e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.06666667014360428f, -3.47693762670076e-9f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.07692307978868484f, -2.8656079731348427e-9f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.09090909361839294f, -2.709302115988521e-9f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.1111111119389534f, -8.278422947149977e-10f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.1428571492433548f, -6.38621200366174e-9f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.20000000298023224f, -2.9802322831784522e-9f));
    p = v41_df_add(v41_df_mul(p, square), vec2(0.3333333432674408f, -9.934107758624577e-9f));
    p = v41_df_add(v41_df_mul(p, square), vec2(1.0f, 0.0f));
    vec2 result = v41_df_mul(v41_df_mul(u, p), vec2(2.0f, 0.0f));
    result = v41_df_add(result, v41_df_mul(vec2(float(e), 0.0f),
        vec2(0.693147182464599609375f, -1.904654323148236e-9f)));
    return result.x + result.y;
}

float v41_scalar_log1p_positive(float x) {
    if (x==0.0f) return x;
    return v41_df_log_positive(v41_df_add(vec2(1.0f,0.0f),vec2(x,0.0f)));
}

#include "fp32_v41_cpu_log_table.glsl"

// Native scalar CPU log contract: a nearby 1+i/128 center, a short
// log(1-r) polynomial, and independently rounded high/low exponent terms.
float v41_cpu_scalar_log_positive(float x) {
    precise float near=x-1.0;
    if (abs(near)<0.0625) {
        precise float ratio=fp32_div(near,2.0+near);
        precise float cancellation=near*ratio;
        precise float twice=ratio+ratio;
        precise float square=twice*twice;
        precise float cube=twice*square;
        precise float polynomial=fp32_fma_exact(0.0125,square,0.0833333333333333333);
        precise float correction=fp32_fma_exact(cube,polynomial,-cancellation);
        return near+correction;
    }
    uint bits=floatBitsToUint(x),mantissa=bits&0x7fffffu;
    float exponent=float(int(bits>>23u)-127);
    uint index=(mantissa+0x8000u)>>16u;
    uvec3 data=v41_cpu_log_centers[index];
    float center=uintBitsToFloat(0x3f000000u|(index<<16u));
    float normalized=uintBitsToFloat(0x3f000000u|mantissa);
    precise float difference=center-normalized;
    precise float reduced=difference*uintBitsToFloat(data.z);
    precise float polynomial=fp32_fma_exact(reduced,0.3333333333333333333,0.5);
    precise float square=reduced*reduced;
    precise float term=fp32_fma_exact(polynomial,square,reduced);
    precise float low=fp32_fma_exact(exponent,uintBitsToFloat(v41_cpu_log_centers[128].y),-term);
    low+=uintBitsToFloat(data.y);
    precise float high=fp32_fma_exact(exponent,uintBitsToFloat(v41_cpu_log_centers[128].x),uintBitsToFloat(data.x));
    return high+low;
}

// Windows scalar log1pf rounds 1+x before logf, then subtracts the
// separately rounded cancellation correction. Preserve that operation order.
float v41_cpu_scalar_log1p_positive(float x) {
    precise float rounded=1.0f+x;
    precise float lost=rounded-1.0f;
    lost-=x;
    precise float correction=fp32_div(lost,rounded);
    precise float logarithm=v41_cpu_scalar_log_positive(rounded);
    return logarithm-correction;
}

// Positive-input adaptation of SLEEF xlog1pf, revision
// 60e76d2bce17d278b439d9da17177c8f957a9e9b (Boost Software License 1.0).
// Keep its compensated division and polynomial staging for the CPU vector body.
float v41_sleef_log1p_positive(float d) {
    precise float dp1 = d + 1.0f;
    precise float scaled = dp1 * 1.33333333333333333333f;
    int e = int(floatBitsToUint(scaled) >> 23u) - 127;
    float scale = ldexp(1.0f, -e);
    precise float m = fp32_fma_exact(d, scale, scale - 1.0f);
    precise float shi = 0.693147182464599609375f * float(e);
    precise float slo = fp32_fma_exact(0.693147182464599609375f, float(e), -shi);
    slo = fp32_fma_exact(-1.904654323148236e-9f, float(e), slo);
    vec2 denominator = v41_df_add(vec2(2.0f, 0.0f), vec2(m, 0.0f));
    precise float reciprocal = fp32_div(1.0f, denominator.x);
    precise float hi = m * reciprocal;
    precise float error = fp32_fma_exact(reciprocal, m, -hi);
    precise float correction = fp32_fma_exact(-denominator.x, reciprocal, 1.0f);
    correction = fp32_fma_exact(-denominator.y, reciprocal, correction);
    precise float lo = fp32_fma_exact(0.0f, reciprocal, error);
    lo = fp32_fma_exact(hi, correction, lo);
    precise float x2 = hi * hi;
    precise float p = 0.3027294874f;
    p = fp32_fma_exact(p, x2, 0.3996108174f);
    p = fp32_fma_exact(p, x2, 0.6666694880f);
    vec2 result = v41_df_add(vec2(shi, slo), vec2(hi * 2.0f, lo * 2.0f));
    precise float tail = (x2 * hi) * p;
    result = v41_df_add(result, vec2(tail, 0.0f));
    return result.x + result.y;
}

float v41_cpu_softplus(float x, uint index, uint elements) {
    if (x > 20.0f) return x;
    bool vector_body = index < elements / 32u * 32u;
    float e = vector_body ? sleef_expf_u10(x) : v41_scalar_exp(x);
    return vector_body ? v41_sleef_log1p_positive(e) : v41_cpu_scalar_log1p_positive(e);
}


float v41_sqrt(float x) {
    uint xb = floatBitsToUint(x);
    if (x <= 0.0f || isinf(x) || isnan(x) || xb < 0x00800000u) return sqrt(x);
    uint sb = floatBitsToUint(sqrt(x));
    while (fp32_square_compare(sb, xb, false) > 0) --sb;
    while (fp32_square_compare(sb + 1u, xb, false) <= 0) ++sb;
    int cmp = fp32_square_compare(sb, xb, true);
    if (cmp < 0 || (cmp == 0 && (sb & 1u) != 0u)) ++sb;
    return uintBitsToFloat(sb);
}

