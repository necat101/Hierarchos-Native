"""Freeze CPU non-last-axis ratio-2 pooling at SIMD/tail channel widths.

The expression is the released Compressor pool: reshape rows into two-slot
groups, softmax the slot axis, multiply KV, then sum the slots. Independent
FP32 PyTorch autograd pins both input VJPs without projection/norm roundoff.
"""
import json
from pathlib import Path
import torch
from validate_deepseek_v41_compressor_decode_grad import deterministic,bits

FIXTURE=Path(__file__).parent/"fixtures/deepseek_v41_compressor_pool_math.json"

def make_fixture():
    torch.set_num_threads(1)
    cases=[]
    for i,width in enumerate([1,3,4,7,8,15,16,17,31,32,33,65]):
        kv=deterministic((2,2,2,width),7000+i*11,1.0).requires_grad_()
        score=deterministic((2,2,2,width),7001+i*11,6.0).requires_grad_()
        upstream=deterministic((2,2,width),7002+i*11,0.3)
        pooled=(kv*score.softmax(dim=2)).sum(dim=2)
        (pooled*upstream).sum().backward()
        cases.append(dict(name=f"channels-{width}",batch=2,seq=4,dim=width,
            kv_bits=bits(kv),score_bits=bits(score),upstream_bits=bits(upstream),
            pooled_bits=bits(pooled),grad_kv_bits=bits(kv.grad),grad_score_bits=bits(score.grad)))
    return dict(torch_version=torch.__version__,torch_revision=torch.version.git_version,
        tolerance=2e-7,rtol=0.0,comparison="strict_less_than",cases=cases)

if __name__=="__main__":
    result=make_fixture()
    if FIXTURE.exists() and json.loads(FIXTURE.read_text())!=result:
        raise SystemExit("compressor pool math fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")
    values=sum(len(c[k]) for c in result['cases'] for k in ['pooled_bits','grad_kv_bits','grad_score_bits'])
    print(f"PASS: {len(result['cases'])} SIMD/tail widths, {values} frozen pool/VJP values")
