// --- Exact FP32 fused multiply-add (no shaderFloat64) -----------------------
//
// Gen9-class hardware has no IEEE fused multiply-add: its MAD instruction
// truncates the intermediate product, so IGC lowers GLSL's fma() into a
// separately rounded multiply and add.  The CPU oracle runs the SLEEF
// pipelines on AVX2 with FMA3, i.e. with a genuinely fused multiply-add, and
// a truncated MAD breaks the compensated (two-float) SLEEF staging outright:
// fma(a, b, -ab) is supposed to return the exact product error.
// The fused sequence is therefore rebuilt from exactly rounded pieces:
//
//   product + error == a * b   exactly (Dekker two-product, split 2^12 + 1)
//   sum + error     == p + c   exactly (Knuth two-sum, no ordering assumption)
//   exact value     == sum + (sum_error + product_error)
//
// so `sum + (sum_error + product_error)` is the correctly rounded a * b + c for
// every argument these kernels evaluate.  `precise` disables reassociation
// and contraction because every intermediate step relies on its own rounding.
float fp32_fma_exact(float a, float b, float c) {
    precise float product = a * b;
    precise float scaled_a = a * 4097.0;
    precise float a_high = scaled_a - (scaled_a - a);
    precise float a_low = a - a_high;
    precise float scaled_b = b * 4097.0;
    precise float b_high = scaled_b - (scaled_b - b);
    precise float b_low = b - b_high;
    precise float product_error =
        (((a_high * b_high - product) + a_high * b_low) + a_low * b_high) + a_low * b_low;

    precise float sum = product + c;
    precise float c_virtual = sum - product;
    precise float product_virtual = sum - c_virtual;
    precise float sum_error = (product - product_virtual) + (c - c_virtual);

    return sum + (sum_error + product_error);
}
