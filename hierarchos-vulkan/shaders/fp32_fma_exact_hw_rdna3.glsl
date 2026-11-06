#ifndef HIERARCHOS_FP32_FMA_EXACT_GLSL
#define HIERARCHOS_FP32_FMA_EXACT_GLSL
// --- Exact FP32 fused multiply-add for RDNA3 ---------------------------------
//
// RDNA3's `fma` is an IEEE fused multiply-add: the product is kept exact and
// the final result is rounded once. The committed Gen9 construction in
// fp32_fma_exact.glsl reconstructs that same correctly rounded value from
// Dekker/Knuth error-free transforms because Intel Gen9's MAD truncates the
// intermediate product. Supplying this header first therefore substitutes one
// hardware instruction for the software construction without changing a single
// result bit; the frozen-corpus tests compare raw words on the physical part.
//
// `precise` forbids reassociation and contraction around the call, preserving
// each caller's explicit rounding points.
float fp32_fma_exact(float a, float b, float c) {
    precise float value = fma(a, b, c);
    return value;
}
#endif
