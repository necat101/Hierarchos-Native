#version 450
#extension GL_GOOGLE_include_directive : enable
#line 1 "validation/phase_xi_channel_tiling_20261125/shaders/transformer_attention_backward_key_chantile_rdna3.comp"

#extension GL_GOOGLE_include_directive : require
#line 1 "shaders/fp32_fma_exact.glsl"



















float fp32_fma_exact(float a, float b, float c) {
    precise float product = a * b;
    precise float scaled_a = a * 4097.0;
    precise float a_high = scaled_a - (scaled_a - a);
    precise float a_low = a - a_high;
    precise float scaled_b = b * 4097.0;
    precise float b_high = scaled_b - (scaled_b - b);
    precise float b_low = b - b_high;
    precise float product_error =
        ( ( (a_high * b_high - product) + a_high * b_low) + a_low * b_high) + a_low * b_low;

    precise float sum = product + c;
    precise float c_virtual = sum - product;
    precise float product_virtual = sum - c_virtual;
    precise float sum_error = (product - product_virtual) + (c - c_virtual);

    return sum + (sum_error + product_error);
}


#line 4 "validation/phase_xi_channel_tiling_20261125/shaders/transformer_attention_backward_key_chantile_rdna3.comp"
#line 1 "shaders/fp32_sqrt_recip.glsl"




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
    if (a.z != b.z) return a.z < b.z ? - 1 : 1;
    if (a.y != b.y) return a.y < b.y ? - 1 : 1;
    return a.x == b.x ? 0 : (a.x < b.x ? - 1 : 1);
}

uvec3 fp32_shift_one96(int shift) {
    if (shift < 32) return uvec3(1u << uint(shift), 0u, 0u);
    if (shift < 64) return uvec3(0u, 1u << uint(shift - 32), 0u);
    return uvec3(0u, 0u, 1u << uint(shift - 64));
}
uvec2 fp32_shift(uint a, int shift) {
    if (shift == 0) return uvec2(a, 0u);
    if (shift < 32) return uvec2(a << uint(shift), a >> uint(32 - shift));
    return uvec2(0u, a << uint(shift - 32));
}
int fp32_compare(uvec2 a, uvec2 b) {
    if (a.y != b.y) return a.y < b.y ? - 1 : 1;
    return a.x == b.x ? 0 : (a.x < b.x ? - 1 : 1);
}
uint fp32_mantissa(uint bits) { return(bits & 0x7fffffu) | 0x800000u; }
int fp32_exponent(uint bits) { return int(bits >> 23u) - 127; }
int fp32_square_compare(uint candidate, uint input_bits, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = fp32_exponent(input_bits) - 2 * fp32_exponent(candidate) + 23;
    if (midpoint) { m = 2u * m + 1u; shift += 2; }
    return fp32_compare(fp32_product(m, m), fp32_shift(fp32_mantissa(input_bits), shift));
}
int fp32_recip_compare(uint candidate, uint denominator, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = 46 - fp32_exponent(candidate) - fp32_exponent(denominator);
    if (midpoint) { m = 2u * m + 1u; shift += 1; }
    return fp32_compare(fp32_product(m, fp32_mantissa(denominator)), fp32_shift(1u, shift));
}
int fp32_div_compare(uint candidate, uint numerator, uint denominator, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = fp32_exponent(numerator) - fp32_exponent(candidate) - fp32_exponent(denominator) + 23;
    if (midpoint) { m = 2u * m + 1u; ++ shift; }
    return fp32_compare(fp32_product(m, fp32_mantissa(denominator)), fp32_shift(fp32_mantissa(numerator), shift));
}
float fp32_div(float numerator, float denominator) {
    float approximate = numerator / denominator;
    uint nb = floatBitsToUint(abs(numerator)), db = floatBitsToUint(abs(denominator));
    uint qb = floatBitsToUint(abs(approximate));
    if (nb < 0x00800000u || db < 0x00800000u || qb < 0x00800000u ||
        nb >= 0x7f800000u || db >= 0x7f800000u || qb >= 0x7f800000u) return approximate;
    while (fp32_div_compare(qb, nb, db, false) > 0) -- qb;
    while (fp32_div_compare(qb + 1u, nb, db, false) <= 0) ++ qb;
    int cmp = fp32_div_compare(qb, nb, db, true);
    if (cmp < 0 || (cmp == 0 && (qb & 1u) != 0u)) ++ qb;
    return uintBitsToFloat(qb | ( (floatBitsToUint(numerator) ^ floatBitsToUint(denominator)) & 0x80000000u));
}




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
    if (shift >= 96) return - 1;
    return fp32_compare96(product, fp32_shift_one96(shift));
}








float fp32_rsqrt(float x) {
    uint xb = floatBitsToUint(x);
    if (x <= 0.0 || isinf(x) || isnan(x) || xb < 0x00800000u)
        return inversesqrt(x);
    uint rb = floatBitsToUint(inversesqrt(x));
    if (rb < 0x00800000u || rb >= 0x7f800000u) return uintBitsToFloat(rb);
    while (fp32_rsqrt_compare(rb, xb, false) > 0) -- rb;
    while (fp32_rsqrt_compare(rb + 1u, xb, false) <= 0) ++ rb;
    int cmp = fp32_rsqrt_compare(rb, xb, true);
    if (cmp < 0 || (cmp == 0 && (rb & 1u) != 0u)) ++ rb;
    return uintBitsToFloat(rb);
}

float fp32_sqrt_recip(float x) {
    uint xb = floatBitsToUint(x);
    if (x <= 0.0 || isinf(x) || isnan(x) || xb < 0x00800000u)
        return 1.0 / sqrt(x);
    uint sb = floatBitsToUint(sqrt(x));
    while (fp32_square_compare(sb, xb, false) > 0) -- sb;
    while (fp32_square_compare(sb + 1u, xb, false) <= 0) ++ sb;
    int cmp = fp32_square_compare(sb, xb, true);
    if (cmp < 0 || (cmp == 0 && (sb & 1u) != 0u)) ++ sb;
    uint rb = floatBitsToUint(1.0 / uintBitsToFloat(sb));
    while (fp32_recip_compare(rb, sb, false) > 0) -- rb;
    while (fp32_recip_compare(rb + 1u, sb, false) <= 0) ++ rb;
    cmp = fp32_recip_compare(rb, sb, true);
    if (cmp < 0 || (cmp == 0 && (rb & 1u) != 0u)) ++ rb;
    return uintBitsToFloat(rb);
}















uvec4 fp32_pow_wide_add(uvec4 acc, uvec2 value, int offset) {
    uint words[4] = uint[4] (acc.x, acc.y, acc.z, acc.w);
    uint carry = 0u;
    for (int k = 0; k < 2; ++ k) {
        int index = offset + k;
        if (index > 3) break;
        uint sum = words[index] + (k == 0 ? value.x : value.y);
        uint product_carry = uint(sum < words[index]);
        uint rounded = sum + carry;
        carry = product_carry + uint(rounded < sum);
        words[index] = rounded;
    }
    for (int index = offset + 2; index <= 3 && carry != 0u; ++ index) {
        uint sum = words[index] + carry;
        carry = uint(sum < words[index]);
        words[index] = sum;
    }
    return uvec4(words[0], words[1], words[2], words[3]);
}


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
    if (a.w != b.w) return a.w < b.w ? - 1 : 1;
    if (a.z != b.z) return a.z < b.z ? - 1 : 1;
    if (a.y != b.y) return a.y < b.y ? - 1 : 1;
    return a.x == b.x ? 0 : (a.x < b.x ? - 1 : 1);
}

uvec4 fp32_shift_one128(int shift) {
    if (shift < 32) return uvec4(1u << uint(shift), 0u, 0u, 0u);
    if (shift < 64) return uvec4(0u, 1u << uint(shift - 32), 0u, 0u);
    if (shift < 96) return uvec4(0u, 0u, 1u << uint(shift - 64), 0u);
    return uvec4(0u, 0u, 0u, 1u << uint(shift - 96));
}




int fp32_pow_neg_three_halves_compare(uint candidate, uint input_bits, bool midpoint) {
    uint m = fp32_mantissa(candidate);
    int shift = 115 - 2 * fp32_exponent(candidate) - 3 * fp32_exponent(input_bits);
    if (midpoint) {
        m = 2u * m + 1u;
        shift += 2;
    }
    if (shift < 0) return 1;
    if (shift >= 128) return - 1;
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
    for (int step = 0; step < 8; ++ step) {
        if (fp32_pow_neg_three_halves_compare(cb, xb, false) <= 0) break;
        -- cb;
    }
    for (int step = 0; step < 8; ++ step) {
        if (fp32_pow_neg_three_halves_compare(cb + 1u, xb, false) > 0) break;
        ++ cb;
    }
    int cmp = fp32_pow_neg_three_halves_compare(cb, xb, true);
    if (cmp < 0 || (cmp == 0 && (cb & 1u) != 0u)) ++ cb;
    return uintBitsToFloat(cb);
}
#line 5 "validation/phase_xi_channel_tiling_20261125/shaders/transformer_attention_backward_key_chantile_rdna3.comp"






























































layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint MAX_HEAD_DIM = 128u;
const uint CHUNK = uint(32);
const uint CHANNEL_BLOCKS = uint(4);





































































































layout(set = 0, binding = 0, std430) readonly buffer QBuffer { float q_values[]; };
layout(set = 0, binding = 1, std430) readonly buffer GradOutputBuffer { float grad_output[]; };
layout(set = 0, binding = 2, std430) readonly buffer ProbabilityBuffer { float scratch_probability[]; };
layout(set = 0, binding = 3, std430) readonly buffer DsBuffer { float scratch_ds[]; };
layout(set = 0, binding = 4, std430) buffer GradKBuffer { float grad_k[]; };
layout(set = 0, binding = 5, std430) buffer GradVBuffer { float grad_v[]; };
layout(set = 0, binding = 6, std430) readonly buffer KeyMaskBuffer { float key_mask[]; };
layout(set = 0, binding = 7, std430) readonly buffer QueryMaskBuffer { float query_mask[]; };

layout(push_constant) uniform Push {
    uint batch_size;
    uint query_seq_len;
    uint key_value_seq_len;
    uint query_hidden_size;
    uint value_hidden_size;
    uint num_heads;
    uint num_key_value_heads;
    uint window_size;
    float scale;
    float value_scale;
    float softcap;
    uint alibi_mode;
    float alibi_bias_max;
    float dropout_p;
    uint rng_step;
    uint rng_site;
    uint rng_seed;
    uint has_sink;
    uint causal;
    uint relative_bias_mode;
    uint relative_bias_num_buckets;
    uint relative_bias_max_distance;
    uint query_position_offset;
    uint has_sparse_mask;
    uint compressed_key_count;
    uint compressed_key_rate;
    uint paged_kv;
    uint kv_page_size;
    uint kv_page_table_stride;
    uint unfused_qk;
} pc;



bool query_contributes(uint query_row, uint key, uint absolute_query) {
    if (query_mask[query_row] <= 0.0) return false;
    if (pc.causal != 0u && key > absolute_query) return false;
    return true;
}

void main() {
    uint rows = pc.batch_size * pc.num_key_value_heads * pc.key_value_seq_len;
    uint slot_index = gl_GlobalInvocationID.x;




    uint chunk_index = slot_index % CHANNEL_BLOCKS;
    uint row = slot_index / CHANNEL_BLOCKS;
    if (row >= rows) return;

    uint qk_head_dim = pc.query_hidden_size / pc.num_heads;
    uint value_head_dim = pc.value_hidden_size / pc.num_key_value_heads;
    if (qk_head_dim > MAX_HEAD_DIM || value_head_dim > MAX_HEAD_DIM) return;

    uint c0 = chunk_index * CHUNK;



    if (c0 >= qk_head_dim && c0 >= value_head_dim) return;

    uint key_hidden_size = qk_head_dim * pc.num_key_value_heads;
    uint queries_per_kv = pc.num_heads / pc.num_key_value_heads;
    uint output_hidden_size = value_head_dim * pc.num_heads;

    uint key = row % pc.key_value_seq_len;
    uint rest = row / pc.key_value_seq_len;
    uint kv_head = rest % pc.num_key_value_heads;
    uint b = rest / pc.num_key_value_heads;

    uint k_base = (b * pc.key_value_seq_len + key) * key_hidden_size + kv_head * qk_head_dim;
    uint v_base = (b * pc.key_value_seq_len + key) * pc.value_hidden_size
        + kv_head * value_head_dim;

    if (key_mask[b * pc.key_value_seq_len + key] <= 0.0) {
                                  { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } }
                                  { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } }
        return;
    }

    uint first_head = kv_head * queries_per_kv;

    if (pc.unfused_qk != 9u) {



        float dk[32];
        float dv[32];
                                 { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { dk[0] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { dk[1] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { dk[2] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { dk[3] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { dk[4] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { dk[5] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { dk[6] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { dk[7] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { dk[8] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { dk[9] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { dk[10] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { dk[11] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { dk[12] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { dk[13] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { dk[14] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { dk[15] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { dk[16] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { dk[17] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { dk[18] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { dk[19] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { dk[20] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { dk[21] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { dk[22] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { dk[23] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { dk[24] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { dk[25] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { dk[26] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { dk[27] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { dk[28] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { dk[29] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { dk[30] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { dk[31] = 0.0; } }
                                 { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { dv[0] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { dv[1] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { dv[2] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { dv[3] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { dv[4] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { dv[5] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { dv[6] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { dv[7] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { dv[8] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { dv[9] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { dv[10] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { dv[11] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { dv[12] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { dv[13] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { dv[14] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { dv[15] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { dv[16] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { dv[17] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { dv[18] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { dv[19] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { dv[20] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { dv[21] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { dv[22] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { dv[23] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { dv[24] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { dv[25] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { dv[26] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { dv[27] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { dv[28] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { dv[29] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { dv[30] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { dv[31] = 0.0; } }

        for (uint head = first_head; head < first_head + queries_per_kv; ++ head) {
            for (uint query = 0u; query < pc.query_seq_len; ++ query) {
                uint query_row = b * pc.query_seq_len + query;
                uint absolute_query = pc.query_position_offset + query;
                if (! query_contributes(query_row, key, absolute_query)) continue;
                uint slot = (b * pc.num_heads + head) * pc.query_seq_len * pc.key_value_seq_len
                    + query * pc.key_value_seq_len + key;
                float p = scratch_probability[slot];
                float ds = scratch_ds[slot];
                uint grad_out_base = query_row * output_hidden_size + head * value_head_dim;
                uint q_base = query_row * pc.query_hidden_size + head * qk_head_dim;
                                        { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[0] + contribution; dk[0] = next_k; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[1] + contribution; dk[1] = next_k; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[2] + contribution; dk[2] = next_k; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[3] + contribution; dk[3] = next_k; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[4] + contribution; dk[4] = next_k; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[5] + contribution; dk[5] = next_k; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[6] + contribution; dk[6] = next_k; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[7] + contribution; dk[7] = next_k; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[8] + contribution; dk[8] = next_k; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[9] + contribution; dk[9] = next_k; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[10] + contribution; dk[10] = next_k; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[11] + contribution; dk[11] = next_k; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[12] + contribution; dk[12] = next_k; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[13] + contribution; dk[13] = next_k; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[14] + contribution; dk[14] = next_k; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[15] + contribution; dk[15] = next_k; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[16] + contribution; dk[16] = next_k; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[17] + contribution; dk[17] = next_k; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[18] + contribution; dk[18] = next_k; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[19] + contribution; dk[19] = next_k; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[20] + contribution; dk[20] = next_k; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[21] + contribution; dk[21] = next_k; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[22] + contribution; dk[22] = next_k; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[23] + contribution; dk[23] = next_k; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[24] + contribution; dk[24] = next_k; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[25] + contribution; dk[25] = next_k; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[26] + contribution; dk[26] = next_k; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[27] + contribution; dk[27] = next_k; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[28] + contribution; dk[28] = next_k; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[29] + contribution; dk[29] = next_k; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[30] + contribution; dk[30] = next_k; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[31] + contribution; dk[31] = next_k; } }
                                        { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[0] + contribution; dv[0] = next_v; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[1] + contribution; dv[1] = next_v; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[2] + contribution; dv[2] = next_v; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[3] + contribution; dv[3] = next_v; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[4] + contribution; dv[4] = next_v; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[5] + contribution; dv[5] = next_v; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[6] + contribution; dv[6] = next_v; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[7] + contribution; dv[7] = next_v; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[8] + contribution; dv[8] = next_v; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[9] + contribution; dv[9] = next_v; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[10] + contribution; dv[10] = next_v; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[11] + contribution; dv[11] = next_v; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[12] + contribution; dv[12] = next_v; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[13] + contribution; dv[13] = next_v; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[14] + contribution; dv[14] = next_v; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[15] + contribution; dv[15] = next_v; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[16] + contribution; dv[16] = next_v; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[17] + contribution; dv[17] = next_v; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[18] + contribution; dv[18] = next_v; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[19] + contribution; dv[19] = next_v; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[20] + contribution; dv[20] = next_v; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[21] + contribution; dv[21] = next_v; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[22] + contribution; dv[22] = next_v; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[23] + contribution; dv[23] = next_v; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[24] + contribution; dv[24] = next_v; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[25] + contribution; dv[25] = next_v; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[26] + contribution; dv[26] = next_v; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[27] + contribution; dv[27] = next_v; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[28] + contribution; dv[28] = next_v; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[29] + contribution; dv[29] = next_v; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[30] + contribution; dv[30] = next_v; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[31] + contribution; dv[31] = next_v; } }
            }
        }

                                  { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[0]; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[1]; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[2]; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[3]; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[4]; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[5]; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[6]; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[7]; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[8]; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[9]; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[10]; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[11]; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[12]; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[13]; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[14]; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[15]; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[16]; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[17]; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[18]; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[19]; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[20]; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[21]; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[22]; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[23]; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[24]; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[25]; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[26]; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[27]; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[28]; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[29]; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[30]; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[31]; } }
                                  { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[0]; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[1]; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[2]; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[3]; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[4]; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[5]; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[6]; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[7]; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[8]; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[9]; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[10]; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[11]; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[12]; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[13]; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[14]; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[15]; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[16]; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[17]; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[18]; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[19]; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[20]; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[21]; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[22]; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[23]; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[24]; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[25]; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[26]; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[27]; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[28]; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[29]; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[30]; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[31]; } }
        return;
    }





    float dk[32];
    float dv[32];
                             { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { dk[0] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { dk[1] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { dk[2] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { dk[3] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { dk[4] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { dk[5] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { dk[6] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { dk[7] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { dk[8] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { dk[9] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { dk[10] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { dk[11] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { dk[12] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { dk[13] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { dk[14] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { dk[15] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { dk[16] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { dk[17] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { dk[18] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { dk[19] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { dk[20] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { dk[21] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { dk[22] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { dk[23] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { dk[24] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { dk[25] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { dk[26] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { dk[27] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { dk[28] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { dk[29] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { dk[30] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { dk[31] = 0.0; } }
                             { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { dv[0] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { dv[1] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { dv[2] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { dv[3] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { dv[4] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { dv[5] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { dv[6] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { dv[7] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { dv[8] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { dv[9] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { dv[10] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { dv[11] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { dv[12] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { dv[13] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { dv[14] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { dv[15] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { dv[16] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { dv[17] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { dv[18] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { dv[19] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { dv[20] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { dv[21] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { dv[22] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { dv[23] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { dv[24] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { dv[25] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { dv[26] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { dv[27] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { dv[28] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { dv[29] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { dv[30] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { dv[31] = 0.0; } }

    for (uint head = first_head; head < first_head + queries_per_kv; ++ head) {
        float hk[32];
        float hv[32];
                                  { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { hk[0] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { hk[1] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { hk[2] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { hk[3] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { hk[4] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { hk[5] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { hk[6] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { hk[7] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { hk[8] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { hk[9] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { hk[10] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { hk[11] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { hk[12] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { hk[13] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { hk[14] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { hk[15] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { hk[16] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { hk[17] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { hk[18] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { hk[19] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { hk[20] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { hk[21] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { hk[22] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { hk[23] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { hk[24] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { hk[25] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { hk[26] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { hk[27] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { hk[28] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { hk[29] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { hk[30] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { hk[31] = 0.0; } }
                                  { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { hv[0] = 0.0; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { hv[1] = 0.0; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { hv[2] = 0.0; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { hv[3] = 0.0; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { hv[4] = 0.0; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { hv[5] = 0.0; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { hv[6] = 0.0; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { hv[7] = 0.0; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { hv[8] = 0.0; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { hv[9] = 0.0; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { hv[10] = 0.0; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { hv[11] = 0.0; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { hv[12] = 0.0; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { hv[13] = 0.0; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { hv[14] = 0.0; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { hv[15] = 0.0; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { hv[16] = 0.0; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { hv[17] = 0.0; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { hv[18] = 0.0; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { hv[19] = 0.0; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { hv[20] = 0.0; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { hv[21] = 0.0; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { hv[22] = 0.0; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { hv[23] = 0.0; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { hv[24] = 0.0; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { hv[25] = 0.0; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { hv[26] = 0.0; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { hv[27] = 0.0; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { hv[28] = 0.0; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { hv[29] = 0.0; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { hv[30] = 0.0; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { hv[31] = 0.0; } }
        for (uint query = 0u; query < pc.query_seq_len; ++ query) {
            uint query_row = b * pc.query_seq_len + query;
            uint absolute_query = pc.query_position_offset + query;
            if (! query_contributes(query_row, key, absolute_query)) continue;
            uint slot = (b * pc.num_heads + head) * pc.query_seq_len * pc.key_value_seq_len
                + query * pc.key_value_seq_len + key;
            float p = scratch_probability[slot];
            float ds = scratch_ds[slot];
            uint grad_out_base = query_row * output_hidden_size + head * value_head_dim;
            uint q_base = query_row * pc.query_hidden_size + head * qk_head_dim;
            precise float score_scale = pc.scale * 1.0;
            score_scale = score_scale * ds;
                                     { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[0] + contribution; hk[0] = next_k; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[1] + contribution; hk[1] = next_k; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[2] + contribution; hk[2] = next_k; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[3] + contribution; hk[3] = next_k; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[4] + contribution; hk[4] = next_k; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[5] + contribution; hk[5] = next_k; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[6] + contribution; hk[6] = next_k; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[7] + contribution; hk[7] = next_k; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[8] + contribution; hk[8] = next_k; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[9] + contribution; hk[9] = next_k; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[10] + contribution; hk[10] = next_k; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[11] + contribution; hk[11] = next_k; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[12] + contribution; hk[12] = next_k; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[13] + contribution; hk[13] = next_k; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[14] + contribution; hk[14] = next_k; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[15] + contribution; hk[15] = next_k; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[16] + contribution; hk[16] = next_k; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[17] + contribution; hk[17] = next_k; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[18] + contribution; hk[18] = next_k; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[19] + contribution; hk[19] = next_k; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[20] + contribution; hk[20] = next_k; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[21] + contribution; hk[21] = next_k; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[22] + contribution; hk[22] = next_k; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[23] + contribution; hk[23] = next_k; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[24] + contribution; hk[24] = next_k; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[25] + contribution; hk[25] = next_k; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[26] + contribution; hk[26] = next_k; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[27] + contribution; hk[27] = next_k; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[28] + contribution; hk[28] = next_k; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[29] + contribution; hk[29] = next_k; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[30] + contribution; hk[30] = next_k; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[31] + contribution; hk[31] = next_k; } }
                                     { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[0] + product; hv[0] = next_v; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[1] + product; hv[1] = next_v; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[2] + product; hv[2] = next_v; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[3] + product; hv[3] = next_v; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[4] + product; hv[4] = next_v; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[5] + product; hv[5] = next_v; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[6] + product; hv[6] = next_v; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[7] + product; hv[7] = next_v; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[8] + product; hv[8] = next_v; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[9] + product; hv[9] = next_v; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[10] + product; hv[10] = next_v; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[11] + product; hv[11] = next_v; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[12] + product; hv[12] = next_v; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[13] + product; hv[13] = next_v; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[14] + product; hv[14] = next_v; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[15] + product; hv[15] = next_v; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[16] + product; hv[16] = next_v; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[17] + product; hv[17] = next_v; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[18] + product; hv[18] = next_v; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[19] + product; hv[19] = next_v; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[20] + product; hv[20] = next_v; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[21] + product; hv[21] = next_v; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[22] + product; hv[22] = next_v; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[23] + product; hv[23] = next_v; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[24] + product; hv[24] = next_v; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[25] + product; hv[25] = next_v; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[26] + product; hv[26] = next_v; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[27] + product; hv[27] = next_v; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[28] + product; hv[28] = next_v; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[29] + product; hv[29] = next_v; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[30] + product; hv[30] = next_v; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[31] + product; hv[31] = next_v; } }
        }
                                 { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { dk[0] = dk[0] + hk[0]; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { dk[1] = dk[1] + hk[1]; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { dk[2] = dk[2] + hk[2]; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { dk[3] = dk[3] + hk[3]; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { dk[4] = dk[4] + hk[4]; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { dk[5] = dk[5] + hk[5]; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { dk[6] = dk[6] + hk[6]; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { dk[7] = dk[7] + hk[7]; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { dk[8] = dk[8] + hk[8]; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { dk[9] = dk[9] + hk[9]; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { dk[10] = dk[10] + hk[10]; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { dk[11] = dk[11] + hk[11]; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { dk[12] = dk[12] + hk[12]; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { dk[13] = dk[13] + hk[13]; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { dk[14] = dk[14] + hk[14]; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { dk[15] = dk[15] + hk[15]; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { dk[16] = dk[16] + hk[16]; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { dk[17] = dk[17] + hk[17]; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { dk[18] = dk[18] + hk[18]; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { dk[19] = dk[19] + hk[19]; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { dk[20] = dk[20] + hk[20]; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { dk[21] = dk[21] + hk[21]; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { dk[22] = dk[22] + hk[22]; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { dk[23] = dk[23] + hk[23]; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { dk[24] = dk[24] + hk[24]; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { dk[25] = dk[25] + hk[25]; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { dk[26] = dk[26] + hk[26]; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { dk[27] = dk[27] + hk[27]; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { dk[28] = dk[28] + hk[28]; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { dk[29] = dk[29] + hk[29]; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { dk[30] = dk[30] + hk[30]; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { dk[31] = dk[31] + hk[31]; } }
                                 { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { dv[0] = dv[0] + hv[0]; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { dv[1] = dv[1] + hv[1]; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { dv[2] = dv[2] + hv[2]; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { dv[3] = dv[3] + hv[3]; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { dv[4] = dv[4] + hv[4]; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { dv[5] = dv[5] + hv[5]; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { dv[6] = dv[6] + hv[6]; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { dv[7] = dv[7] + hv[7]; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { dv[8] = dv[8] + hv[8]; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { dv[9] = dv[9] + hv[9]; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { dv[10] = dv[10] + hv[10]; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { dv[11] = dv[11] + hv[11]; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { dv[12] = dv[12] + hv[12]; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { dv[13] = dv[13] + hv[13]; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { dv[14] = dv[14] + hv[14]; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { dv[15] = dv[15] + hv[15]; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { dv[16] = dv[16] + hv[16]; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { dv[17] = dv[17] + hv[17]; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { dv[18] = dv[18] + hv[18]; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { dv[19] = dv[19] + hv[19]; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { dv[20] = dv[20] + hv[20]; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { dv[21] = dv[21] + hv[21]; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { dv[22] = dv[22] + hv[22]; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { dv[23] = dv[23] + hv[23]; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { dv[24] = dv[24] + hv[24]; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { dv[25] = dv[25] + hv[25]; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { dv[26] = dv[26] + hv[26]; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { dv[27] = dv[27] + hv[27]; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { dv[28] = dv[28] + hv[28]; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { dv[29] = dv[29] + hv[29]; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { dv[30] = dv[30] + hv[30]; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { dv[31] = dv[31] + hv[31]; } }
    }

                              { const uint d_ = c0 + uint(0); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[0]; } } { const uint d_ = c0 + uint(1); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[1]; } } { const uint d_ = c0 + uint(2); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[2]; } } { const uint d_ = c0 + uint(3); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[3]; } } { const uint d_ = c0 + uint(4); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[4]; } } { const uint d_ = c0 + uint(5); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[5]; } } { const uint d_ = c0 + uint(6); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[6]; } } { const uint d_ = c0 + uint(7); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[7]; } } { const uint d_ = c0 + uint(8); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[8]; } } { const uint d_ = c0 + uint(9); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[9]; } } { const uint d_ = c0 + uint(10); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[10]; } } { const uint d_ = c0 + uint(11); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[11]; } } { const uint d_ = c0 + uint(12); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[12]; } } { const uint d_ = c0 + uint(13); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[13]; } } { const uint d_ = c0 + uint(14); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[14]; } } { const uint d_ = c0 + uint(15); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[15]; } } { const uint d_ = c0 + uint(16); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[16]; } } { const uint d_ = c0 + uint(17); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[17]; } } { const uint d_ = c0 + uint(18); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[18]; } } { const uint d_ = c0 + uint(19); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[19]; } } { const uint d_ = c0 + uint(20); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[20]; } } { const uint d_ = c0 + uint(21); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[21]; } } { const uint d_ = c0 + uint(22); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[22]; } } { const uint d_ = c0 + uint(23); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[23]; } } { const uint d_ = c0 + uint(24); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[24]; } } { const uint d_ = c0 + uint(25); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[25]; } } { const uint d_ = c0 + uint(26); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[26]; } } { const uint d_ = c0 + uint(27); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[27]; } } { const uint d_ = c0 + uint(28); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[28]; } } { const uint d_ = c0 + uint(29); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[29]; } } { const uint d_ = c0 + uint(30); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[30]; } } { const uint d_ = c0 + uint(31); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[31]; } }
                              { const uint d_ = c0 + uint(0); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[0]; } } { const uint d_ = c0 + uint(1); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[1]; } } { const uint d_ = c0 + uint(2); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[2]; } } { const uint d_ = c0 + uint(3); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[3]; } } { const uint d_ = c0 + uint(4); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[4]; } } { const uint d_ = c0 + uint(5); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[5]; } } { const uint d_ = c0 + uint(6); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[6]; } } { const uint d_ = c0 + uint(7); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[7]; } } { const uint d_ = c0 + uint(8); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[8]; } } { const uint d_ = c0 + uint(9); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[9]; } } { const uint d_ = c0 + uint(10); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[10]; } } { const uint d_ = c0 + uint(11); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[11]; } } { const uint d_ = c0 + uint(12); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[12]; } } { const uint d_ = c0 + uint(13); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[13]; } } { const uint d_ = c0 + uint(14); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[14]; } } { const uint d_ = c0 + uint(15); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[15]; } } { const uint d_ = c0 + uint(16); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[16]; } } { const uint d_ = c0 + uint(17); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[17]; } } { const uint d_ = c0 + uint(18); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[18]; } } { const uint d_ = c0 + uint(19); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[19]; } } { const uint d_ = c0 + uint(20); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[20]; } } { const uint d_ = c0 + uint(21); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[21]; } } { const uint d_ = c0 + uint(22); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[22]; } } { const uint d_ = c0 + uint(23); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[23]; } } { const uint d_ = c0 + uint(24); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[24]; } } { const uint d_ = c0 + uint(25); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[25]; } } { const uint d_ = c0 + uint(26); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[26]; } } { const uint d_ = c0 + uint(27); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[27]; } } { const uint d_ = c0 + uint(28); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[28]; } } { const uint d_ = c0 + uint(29); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[29]; } } { const uint d_ = c0 + uint(30); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[30]; } } { const uint d_ = c0 + uint(31); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[31]; } }
}
