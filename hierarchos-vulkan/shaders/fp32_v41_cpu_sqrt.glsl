// Pinned CPU VML sqrt contract, observed independently on AMD Zen 4.
// The complete 8192-entry SSE seed map is indexed by exponent parity and
// the upper twelve mantissa bits. No model values or weights are in it.
// V41_SQRT_SEEDS names a read-only uint storage buffer supplied by the caller.
float v41_cpu_sqrt(float x) {
    uint bits=floatBitsToUint(x);
    if (x<=0.0 || isnan(x) || isinf(x) || bits<0x00800000u) return v41_sqrt(x);
    int exponent=int(bits>>23u)-127;
    int parity=exponent&1;
    uint index=(bits&0x7fffffu)/2048u+uint(parity)*4096u;
    int half_exponent=(exponent-parity)/2;
    precise float seed=uintBitsToFloat(uint(int(V41_SQRT_SEEDS[index])-half_exponent*8388608));
    precise float root=x*seed;
    precise float half_inverse=seed*0.5;
    precise float product=root*half_inverse;
    precise float correction=0.5-product;
    precise float root_delta=root*correction;
    precise float half_delta=half_inverse*correction;
    root+=root_delta;
    half_inverse+=half_delta;
    precise float square=root*root;
    precise float residual=x-square;
    precise float delta=residual*half_inverse;
    return root+delta;
}
