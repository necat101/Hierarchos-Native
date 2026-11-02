// The scalar CPU cascade has 32 independent accumulation chains. Each lane
// owns one chain; only the final eight-bank fold belongs to lane zero. Call
// collectively from a 32-lane workgroup, including widths with vector tails.
shared float v41_square_banks[32];
shared float v41_square_sum;
float v41_sum_squares(uint base, uint width) {
    uint lane = gl_LocalInvocationID.x;
    precise float acc[4];
    for (uint level = 0u; level < 4u; ++level) acc[level] = 0.0;
    uint vectors = width / 8u, groups = vectors / 4u;
    uint bits = groups > 1u ? uint(findMSB(groups - 1u)) + 1u : 0u;
    uint power = max(4u, bits / 4u), step = 1u << power, mask = step - 1u;
    for (uint group = 0u; group < groups; ++group) {
        float value = V41_REDUCE_INPUT[base + group * 32u + lane];
        precise float square = value * value;
        acc[0] += square;
        uint count = group + 1u;
        if (count % step == 0u) {
            for (uint level = 1u; level < 4u; ++level) {
                acc[level] += acc[level - 1u];
                acc[level - 1u] = 0.0;
                if ((count & (mask << (level * power))) != 0u) break;
            }
        }
    }
    for (uint level = 1u; level < 4u; ++level) acc[0] += acc[level];
    if (lane < 8u) {
        for (uint v = groups * 4u; v < vectors; ++v) {
            float value = V41_REDUCE_INPUT[base + v * 8u + lane];
            precise float square = value * value;
            acc[0] += square;
        }
    }
    v41_square_banks[lane] = acc[0];
    barrier();
    if (lane == 0u) {
        precise float sum = 0.0;
        for (uint col = vectors * 8u; col < width; ++col) {
            float value = V41_REDUCE_INPUT[base + col];
            precise float square = value * value;
            sum += square;
        }
        for (uint i = 0u; i < 8u; ++i) {
            precise float p = v41_square_banks[i] + v41_square_banks[8u + i];
            p += v41_square_banks[16u + i];
            p += v41_square_banks[24u + i];
            sum += p;
        }
        v41_square_sum = sum;
    }
    barrier();
    return v41_square_sum;
}
