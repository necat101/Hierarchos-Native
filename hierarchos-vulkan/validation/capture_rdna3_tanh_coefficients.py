"""Capture the pinned MKL piecewise polynomial, independently of model inputs.

The DLL fingerprint is mandatory because the table RVA is specific to this
Torch build. Only numeric coefficients are read; no native code is executed.
"""
import argparse,ctypes,hashlib,struct
from pathlib import Path
import torch

LIBRARY_SHA256='ca1ce2ea3f0d8d3e31bbac96d1be81aa01313ef1e7c8f6a98f6a28a0b3008bdb'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('shaders/fp32_tanh_mkl_zen4_rdna3.glsl'))
    a=p.parse_args();path=Path(torch.__file__).parent/'lib/torch_cpu.dll'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=LIBRARY_SHA256:raise ValueError('unqualified Torch DLL')
    library=ctypes.CDLL(str(path))
    coefficients=struct.unpack('<216d',ctypes.string_at(library._handle+0xd212340,216*8))
    text='''// Pinned Windows Torch/MKL tanh coefficients, 27 bins x degree seven.
// Captured by validation/capture_rdna3_tanh_coefficients.py, never from model values.
// DLL SHA256: '''+LIBRARY_SHA256+'''
#extension GL_ARB_gpu_shader_fp64 : require
const double rdna3_mkl_tanh_coeff[216]=double[216](
'''+',\n'.join('    '+','.join(format(v,'.17g')+('lf' if '.' in format(v,'.17g') or 'e' in format(v,'.17g') else '.0lf')
    for v in coefficients[i:i+8]) for i in range(0,216,8))+''');
float rdna3_mkl_tanh(float x) {
    uint bits=floatBitsToUint(x),magnitude=bits&0x7fffffffu;
    if(magnitude>0x7f800000u)return x;
    if(magnitude>=0x41200000u)return uintBitsToFloat((bits&0x80000000u)|0x3f800000u);
    uint truncated=magnitude&0x7fe00000u;
    uint index=truncated>0x3de00000u ? (truncated-0x3de00000u)>>21u : 0u;
    uint base=index*8u;
    precise double z=double(uintBitsToFloat(magnitude));
    precise double result=rdna3_mkl_tanh_coeff[base+7u];
    for(int degree=6;degree>=0;--degree) {
        precise double product=result*z;
        result=product+rdna3_mkl_tanh_coeff[base+uint(degree)];
    }
    float value=float(result);
    return uintBitsToFloat((floatBitsToUint(value)&0x7fffffffu)|(bits&0x80000000u));
}
'''
    a.output.write_text(text,newline='\n')
if __name__=='__main__':main()
