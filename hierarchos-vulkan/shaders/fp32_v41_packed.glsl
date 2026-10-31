// Released V4.1 storage: FP8 E4M3FN, FP4 E2M1 (low nibble first), UE8M0.
// Integer bit construction also preserves subnormal decoded weights on devices
// whose ordinary FP32 multiply flushes denormals. No native FP8/FP4 feature.
uint v41_e4m3_bits(uint b) {
    uint sign = (b & 128u) << 24;
    uint e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0u) {
        if (m == 0u) return sign;
        uint p = uint(findMSB(m));
        return sign | ((p + 118u) << 23) | ((m - (1u << p)) << (23u - p));
    }
    return sign | ((e + 120u) << 23) | (m << 20);
}

uint v41_e2m1_bits(uint b) {
    const uint magnitude[8] = uint[8](0u, 0x3f000000u, 0x3f800000u,
        0x3fc00000u, 0x40000000u, 0x40400000u, 0x40800000u, 0x40c00000u);
    return ((b & 8u) << 28) | magnitude[b & 7u];
}

uint v41_scale_bits(uint b, uint scale) {
    uint sign = b & 0x80000000u, mantissa = b & 0x7fffffu;
    int exponent = int((b >> 23) & 255u);
    if (exponent == 0) return b; // decoded signed zero; other codes are normal
    exponent += int(scale) - 127;
    if (exponent >= 255) return sign | 0x7f800000u;
    if (exponent > 0) return sign | (uint(exponent) << 23) | mantissa;
    uint shift = uint(1 - exponent);
    if (shift > 24u) return sign;
    uint full = mantissa | 0x800000u;
    uint lower = full & ((1u << shift) - 1u);
    uint midpoint = 1u << (shift - 1u);
    uint result = full >> shift;
    if (lower > midpoint || (lower == midpoint && (result & 1u) != 0u)) ++result;
    return sign | result;
}

uint v41_bf16_bits(uint b) {
    return (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
}
