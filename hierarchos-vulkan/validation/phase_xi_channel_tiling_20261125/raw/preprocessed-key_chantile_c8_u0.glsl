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
const uint CHUNK = uint(8);
const uint CHANNEL_BLOCKS = uint(16);





































































































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
                                  for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { grad_k[k_base + d_] = 0.0; } } }
                                  for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { grad_v[v_base + d_] = 0.0; } } }
        return;
    }

    uint first_head = kv_head * queries_per_kv;

    if (pc.unfused_qk != 9u) {



        float dk[8];
        float dv[8];
                                 for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { dk[i_] = 0.0; } } }
                                 for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { dv[i_] = 0.0; } } }

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
                                        for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { precise float contribution = pc.scale * ds * q_values[q_base + d_]; precise float next_k = dk[i_] + contribution; dk[i_] = next_k; } } }
                                        for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { precise float contribution = p * grad_output[grad_out_base + d_]; precise float next_v = dv[i_] + contribution; dv[i_] = next_v; } } }
            }
        }

                                  for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[i_]; } } }
                                  for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[i_]; } } }
        return;
    }





    float dk[8];
    float dv[8];
                             for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { dk[i_] = 0.0; } } }
                             for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { dv[i_] = 0.0; } } }

    for (uint head = first_head; head < first_head + queries_per_kv; ++ head) {
        float hk[8];
        float hv[8];
                                  for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { hk[i_] = 0.0; } } }
                                  for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { hv[i_] = 0.0; } } }
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
                                     for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { precise float contribution = score_scale * q_values[q_base + d_]; precise float next_k = hk[i_] + contribution; hk[i_] = next_k; } } }
                                     for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { precise float weighted_p = p * 1.0; precise float product = weighted_p * grad_output[grad_out_base + d_]; product = product * 1.0; precise float next_v = hv[i_] + product; hv[i_] = next_v; } } }
        }
                                 for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { dk[i_] = dk[i_] + hk[i_]; } } }
                                 for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { dv[i_] = dv[i_] + hv[i_]; } } }
    }

                              for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < qk_head_dim) { grad_k[k_base + d_] = dk[i_]; } } }
                              for (uint i_ = 0u; i_ < CHUNK; ++ i_) { { const uint d_ = c0 + uint(i_); if (d_ < value_head_dim) { grad_v[v_base + d_] = dv[i_]; } } }
}
