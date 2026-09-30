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
