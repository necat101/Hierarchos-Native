// Scalar expf contract of the Windows CPU flash-attention tail (eight keys).
// Reduce in FP64 and evaluate exp(r) on [-ln(2)/2, ln(2)/2]. Fourteen Taylor
// terms leave error below FP64 rounding noise; round once to FP32, like expf.
// Kept separate from the staged FP32 SLEEF/vector exponential.
#extension GL_ARB_gpu_shader_fp64 : require
float rdna3_scalar_expf(float x) {
    if (x < -104.0) return 0.0;
    int n=int(roundEven(double(x)*1.442695040888963407359924681002lf));
    double r=fma(-double(n),0.693147180559945309417232121458lf,double(x));
    precise double p=1.1470745597729724713851697978682e-11lf;
    p=fma(p,r,1.6059043836821614599392377170155e-10lf);
    p=fma(p,r,2.0876756987868098979210090321201e-9lf);
    p=fma(p,r,2.5052108385441718775052108385442e-8lf);
    p=fma(p,r,2.7557319223985890652557319223986e-7lf);
    p=fma(p,r,2.7557319223985890652557319223986e-6lf);
    p=fma(p,r,2.4801587301587301587301587301587e-5lf);
    p=fma(p,r,1.9841269841269841269841269841270e-4lf);
    p=fma(p,r,1.3888888888888888888888888888889e-3lf);
    p=fma(p,r,8.3333333333333333333333333333333e-3lf);
    p=fma(p,r,4.1666666666666666666666666666667e-2lf);
    p=fma(p,r,1.6666666666666666666666666666667e-1lf);
    p=fma(p,r,0.5lf);p=fma(p,r,1.0lf);p=fma(p,r,1.0lf);
    return float(p*packDouble2x32(uvec2(0u,uint(n+1023)<<20u)));
}
