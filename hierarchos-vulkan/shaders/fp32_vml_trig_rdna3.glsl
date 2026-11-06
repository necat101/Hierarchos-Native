// MKL VML HA FP32 sine/cosine on the measured Windows AVX512 host.
// Native RDNA3 FMA performs the FP32 quadrant step and FP64 polynomial.
// Coefficients are the library's degree-nine sine polynomial (also used by
// fp32_vml_cos.glsl); no values from model tensors enter this approximation.
// The polynomial matches the CPU exactly over 0..32 radians, including the
// angle-17 RoPE case that the former |x|<=8 guard sent to SLEEF.
float rdna3_vml_polynomial(double r) {
    precise double square=r*r;
    double p=packDouble2x32(uvec2(0x0e4c7deeu,0x3ec5dbdfu));
    p=fma(p,square,packDouble2x32(uvec2(0xeea73463u,0xbf29f6ffu)));
    p=fma(p,square,packDouble2x32(uvec2(0x3804ca96u,0x3f8110edu)));
    p=fma(p,square,packDouble2x32(uvec2(0xbc836587u,0xbfc55554u)));
    precise double product=p*square;
    return float(fma(product,r,r));
}
double rdna3_vml_reduce(float value,float n) {
    double r=fma(-double(n),packDouble2x32(uvec2(0x54440000u,0x400921fbu)),double(value));
    return fma(-double(n),packDouble2x32(uvec2(0x4c4c6629u,0x3d968c23u)),r);
}
float rdna3_vml_sin(float value) {
    float magic=fma(abs(value),uintBitsToFloat(0x3ea2f983u),-12582912.0);
    precise float n=magic+12582912.0;
    float p=rdna3_vml_polynomial(rdna3_vml_reduce(abs(value),n));
    return uintBitsToFloat(floatBitsToUint(p)^(floatBitsToUint(magic)<<31u)
        ^(floatBitsToUint(value)&0x80000000u));
}
float rdna3_vml_cos(float value) {
    precise float shifted=abs(value)+uintBitsToFloat(0x3fc90fdbu);
    float magic=fma(shifted,uintBitsToFloat(0x3ea2f983u),-12582912.0);
    precise float quadrant=magic+12582912.0;
    precise float n=quadrant-0.5;
    float p=rdna3_vml_polynomial(rdna3_vml_reduce(abs(value),n));
    return uintBitsToFloat(floatBitsToUint(p)^(floatBitsToUint(magic)<<31u));
}
