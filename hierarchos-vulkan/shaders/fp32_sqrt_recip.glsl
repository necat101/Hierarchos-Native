// Match two separately rounded IEEE-754 operations: 1.0f / sqrtf(x).
// GLSL Sqrt and division may be approximate. Compare the neighboring FP32
// significands with exact 32x32 -> 64 integer products, without shaderFloat64.
// The caller supplies a positive normal RMS variance including epsilon.
uvec2 fp32_product(uint a, uint b) {
    uint hi, lo;
    umulExtended(a, b, hi, lo);
    return uvec2(lo, hi);
}

uvec3 fp32_product_64x32(uvec2 a, uint b) {
    uint hi0, lo0, hi1, lo1;
    umulExtended(a.x, b, hi0, lo0);
    umulExtended(a.y, b, hi1, lo1);
    uint middle = hi0 + lo1;
    uint carry = uint(middle < hi0);
    return uvec3(lo0, middle, hi1 + carry);
}

int fp32_compare96(uvec3 a, uvec3 b) {
    if (a.z != b.z) return a.z < b.z ? -1 : 1;
    if (a.y != b.y) return a.y < b.y ? -1 : 1;
    return a.x == b.x ? 0 : (a.x < b.x ? -1 : 1);
}

uvec3 fp32_shift_one96(int shift) {
    if (shift < 32) return uvec3(1u << uint(shift), 0u, 0u);
    if (shift < 64) return uvec3(0u, 1u << uint(shift - 32), 0u);
    return uvec3(0u, 0u, 1u << uint(shift - 64));
}
uvec2 fp32_shift(uint a, int shift) {
    if (shift == 0) return uvec2(a, 0u);
    if (shift < 32) return uvec2(a << uint(shift), a >> uint(32-shift));
    return uvec2(0u, a << uint(shift-32));
}
int fp32_compare(uvec2 a, uvec2 b) {
    if (a.y != b.y) return a.y < b.y ? -1 : 1;
    return a.x == b.x ? 0 : (a.x < b.x ? -1 : 1);
}
uint fp32_mantissa(uint bits) { return (bits & 0x7fffffu) | 0x800000u; }
int fp32_exponent(uint bits) { return int(bits >> 23u) - 127; }
int fp32_square_compare(uint candidate, uint input_bits, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = fp32_exponent(input_bits) - 2*fp32_exponent(candidate) + 23;
    if (midpoint) { m = 2u*m+1u; shift += 2; }
    return fp32_compare(fp32_product(m,m), fp32_shift(fp32_mantissa(input_bits),shift));
}
int fp32_recip_compare(uint candidate, uint denominator, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = 46 - fp32_exponent(candidate) - fp32_exponent(denominator);
    if (midpoint) { m = 2u*m+1u; shift += 1; }
    return fp32_compare(fp32_product(m,fp32_mantissa(denominator)),fp32_shift(1u,shift));
}
int fp32_div_compare(uint candidate, uint numerator, uint denominator, bool midpoint) {
    uint m=fp32_mantissa(candidate);
    int shift=fp32_exponent(numerator)-fp32_exponent(candidate)-fp32_exponent(denominator)+23;
    if (midpoint) { m=2u*m+1u; ++shift; }
    return fp32_compare(fp32_product(m,fp32_mantissa(denominator)),fp32_shift(fp32_mantissa(numerator),shift));
}
float fp32_div(float numerator, float denominator) {
    float approximate=numerator/denominator;
    uint nb=floatBitsToUint(abs(numerator)), db=floatBitsToUint(abs(denominator));
    uint qb=floatBitsToUint(abs(approximate));
    if (nb<0x00800000u || db<0x00800000u || qb<0x00800000u ||
        nb>=0x7f800000u || db>=0x7f800000u || qb>=0x7f800000u) return approximate;
    while (fp32_div_compare(qb,nb,db,false)>0) --qb;
    while (fp32_div_compare(qb+1u,nb,db,false)<=0) ++qb;
    int cmp=fp32_div_compare(qb,nb,db,true);
    if (cmp<0 || (cmp==0 && (qb&1u)!=0u)) ++qb;
    return uintBitsToFloat(qb | ((floatBitsToUint(numerator)^floatBitsToUint(denominator))&0x80000000u));
}

// Compare candidate^2 * input against 1 exactly.  A non-positive result means
// candidate <= 1/sqrt(input).  The midpoint form uses the exact midpoint
// between candidate and its next positive FP32 neighbor for round-to-even.
int fp32_rsqrt_compare(uint candidate, uint input_bits, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = 69 - 2 * fp32_exponent(candidate) - fp32_exponent(input_bits);
    if (midpoint) {
        m = 2u * m + 1u;
        shift += 2;
    }
    uvec2 square = fp32_product(m, m);
    uvec3 product = fp32_product_64x32(square, fp32_mantissa(input_bits));
    if (shift < 0) return 1;
    if (shift >= 96) return -1;
    return fp32_compare96(product, fp32_shift_one96(shift));
}

// Correctly round the real reciprocal square root directly to FP32.  This is
// deliberately different from fp32_sqrt_recip(), which rounds sqrt first and
// then rounds the reciprocal. Keep this helper for qualification surfaces
// whose reference numerics match the single-rounding result.  In particular,
// the current local PyTorch CPU oracle makes torch.pow(x, -0.5) bit-identical
// to torch.rsqrt on Gemma4's strict fixture rows; staged sqrt+division can be
// one ulp lower.
float fp32_rsqrt(float x) {
    uint xb = floatBitsToUint(x);
    if (x <= 0.0 || isinf(x) || isnan(x) || xb < 0x00800000u)
        return inversesqrt(x);
    uint rb = floatBitsToUint(inversesqrt(x));
    if (rb < 0x00800000u || rb >= 0x7f800000u) return uintBitsToFloat(rb);
    while (fp32_rsqrt_compare(rb, xb, false) > 0) --rb;
    while (fp32_rsqrt_compare(rb + 1u, xb, false) <= 0) ++rb;
    int cmp = fp32_rsqrt_compare(rb, xb, true);
    if (cmp < 0 || (cmp == 0 && (rb & 1u) != 0u)) ++rb;
    return uintBitsToFloat(rb);
}

float fp32_sqrt_recip(float x) {
    uint xb = floatBitsToUint(x);
    if (x <= 0.0 || isinf(x) || isnan(x) || xb < 0x00800000u)
        return 1.0 / sqrt(x);
    uint sb = floatBitsToUint(sqrt(x));
    while (fp32_square_compare(sb,xb,false) > 0) --sb;
    while (fp32_square_compare(sb+1u,xb,false) <= 0) ++sb;
    int cmp = fp32_square_compare(sb,xb,true);
    if (cmp < 0 || (cmp == 0 && (sb & 1u) != 0u)) ++sb;
    uint rb = floatBitsToUint(1.0 / uintBitsToFloat(sb));
    while (fp32_recip_compare(rb,sb,false) > 0) --rb;
    while (fp32_recip_compare(rb+1u,sb,false) <= 0) ++rb;
    cmp = fp32_recip_compare(rb,sb,true);
    if (cmp < 0 || (cmp == 0 && (rb & 1u) != 0u)) ++rb;
    return uintBitsToFloat(rb);
}

// --- Correctly rounded x^(-3/2) -------------------------------------------
// torch.pow(mean_squared, -1.5) is the derivative factor of the forward
// torch.pow(mean_squared, -0.5).  Its CPU kernel takes the scalar path on the
// tiny strict-fixture tensors, so the reference value is the correctly rounded
// double-precision result; paraphrasing it with SLEEF's u10 log/exp staging
// leaves a few percent of rows one ulp away, and the RMS backward amplifies
// that seed through every lower norm.
//
//   candidate <= x^(-3/2)   <=>   candidate^2 * x^3 <= 1
//
// Both sides are integer mantissas, so the comparison needs only exact 32-bit
// integer products: no fused multiply-add (Gen9's MAD truncates the product,
// which breaks SLEEF's compensated two-float staging outright) and no
// shaderFloat64.
uvec4 fp32_pow_wide_add(uvec4 acc, uvec2 value, int offset) {
    uint words[4] = uint[4](acc.x, acc.y, acc.z, acc.w);
    uint carry = 0u;
    for (int k = 0; k < 2; ++k) {
        int index = offset + k;
        if (index > 3) break;
        uint sum = words[index] + (k == 0 ? value.x : value.y);
        uint product_carry = uint(sum < words[index]);
        uint rounded = sum + carry;
        carry = product_carry + uint(rounded < sum);
        words[index] = rounded;
    }
    for (int index = offset + 2; index <= 3 && carry != 0u; ++index) {
        uint sum = words[index] + carry;
        carry = uint(sum < words[index]);
        words[index] = sum;
    }
    return uvec4(words[0], words[1], words[2], words[3]);
}

// Exact mantissa product of a (< 2^50) and b (< 2^72); the result is < 2^122.
uvec4 fp32_pow_mantissa_product(uvec2 a, uvec3 b) {
    uvec4 acc = uvec4(0u);
    acc = fp32_pow_wide_add(acc, fp32_product(a.x, b.x), 0);
    acc = fp32_pow_wide_add(acc, fp32_product(a.x, b.y), 1);
    acc = fp32_pow_wide_add(acc, fp32_product(a.y, b.x), 1);
    acc = fp32_pow_wide_add(acc, fp32_product(a.x, b.z), 2);
    acc = fp32_pow_wide_add(acc, fp32_product(a.y, b.y), 2);
    acc = fp32_pow_wide_add(acc, fp32_product(a.y, b.z), 3);
    return acc;
}

int fp32_compare128(uvec4 a, uvec4 b) {
    if (a.w != b.w) return a.w < b.w ? -1 : 1;
    if (a.z != b.z) return a.z < b.z ? -1 : 1;
    if (a.y != b.y) return a.y < b.y ? -1 : 1;
    return a.x == b.x ? 0 : (a.x < b.x ? -1 : 1);
}

uvec4 fp32_shift_one128(int shift) {
    if (shift < 32) return uvec4(1u << uint(shift), 0u, 0u, 0u);
    if (shift < 64) return uvec4(0u, 1u << uint(shift - 32), 0u, 0u);
    if (shift < 96) return uvec4(0u, 0u, 1u << uint(shift - 64), 0u);
    return uvec4(0u, 0u, 0u, 1u << uint(shift - 96));
}

// Sign of candidate - x^(-3/2); positive normal operands only.  The midpoint
// form compares against the exact midpoint so round-to-even is resolved
// without any floating-point arithmetic.
int fp32_pow_neg_three_halves_compare(uint candidate, uint input_bits, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = 115 - 2 * fp32_exponent(candidate) - 3 * fp32_exponent(input_bits);
    if (midpoint) {
        m = 2u * m + 1u;
        shift += 2;
    }
    if (shift < 0) return 1;
    if (shift >= 128) return -1;
    uint x_mantissa = fp32_mantissa(input_bits);
    uvec2 square = fp32_product(m, m);
    uvec3 cube = fp32_product_64x32(fp32_product(x_mantissa, x_mantissa), x_mantissa);
    return fp32_compare128(fp32_pow_mantissa_product(square, cube), fp32_shift_one128(shift));
}

float fp32_pow_neg_three_halves(float x) {
    uint xb = floatBitsToUint(x);
    if (x <= 0.0 || isinf(x) || isnan(x) || xb < 0x00800000u) {
        return 1.0 / (x * sqrt(x));
    }
    uint cb = floatBitsToUint(1.0 / (x * sqrt(x)));
    if (cb < 0x00800000u || cb >= 0x7f800000u) return uintBitsToFloat(cb);
    for (int step = 0; step < 8; ++step) {
        if (fp32_pow_neg_three_halves_compare(cb, xb, false) <= 0) break;
        --cb;
    }
    for (int step = 0; step < 8; ++step) {
        if (fp32_pow_neg_three_halves_compare(cb + 1u, xb, false) > 0) break;
        ++cb;
    }
    int cmp = fp32_pow_neg_three_halves_compare(cb, xb, true);
    if (cmp < 0 || (cmp == 0 && (cb & 1u) != 0u)) ++cb;
    return uintBitsToFloat(cb);
}
