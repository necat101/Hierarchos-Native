// Division by a positive power of two is exactly a binary exponent shift.
// Keep the software divider for non-normal operands/results and for scales
// whose reciprocal is not normal, preserving its exceptional-value policy.
float v41_div_scale(float value, float scale) {
    uint sb = floatBitsToUint(scale);
    uint exponent = sb >> 23u;
    if ((sb & 0x807fffffu) == 0u && exponent > 0u && exponent < 254u) {
        precise float quotient = value * uintBitsToFloat((254u - exponent) << 23u);
        uint vb = floatBitsToUint(abs(value));
        uint qb = floatBitsToUint(abs(quotient));
        if (vb >= 0x00800000u && vb < 0x7f800000u &&
            qb >= 0x00800000u && qb < 0x7f800000u) return quotient;
    }
    return fp32_div(value, scale);
}
