"""Pin scalar exp and AVX512 sigmoid body/tail cases independently of native code.

Scalar exp is rounded from Python's float64 libm result; sigmoid is the actual
pinned PyTorch operation, including its vector body and scalar tail. These are
component tests, not the complete-logit acceptance policy.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

FIXTURE=Path(__file__).parent/"fixtures"/"deepseek_v41_unary.json"
TORCH_REVISION="5c4886908584029761b579af026dcfb627c84070"


def make_fixture():
    if torch.version.git_version!=TORCH_REVISION or torch._C._get_cpu_capability()!="AVX512":
        raise RuntimeError("unary reference requires the pinned PyTorch AVX512 operation shapes")
    torch.set_num_threads(1)
    rng=np.random.default_rng(4141027)
    cases=[]
    x=rng.uniform(-80,80,4096).astype(np.float32)
    points=[]
    for k in range(-100,101):
        for offset in [0.0,0.5]:
            p=np.float32((k+offset)*math.log(2))
            points.extend([np.nextafter(p,np.float32(-np.inf)),p,np.nextafter(p,np.float32(np.inf))])
    x=np.concatenate((x,np.asarray(points,dtype=np.float32)))
    y=np.array([math.exp(float(v)) for v in x],dtype=np.float32)
    cases.append(dict(name="scalar-exp-normal-domain-and-range-boundaries",op=0,inputs=x.tolist(),expected_bits=y.view(np.uint32).tolist()))
    for n in [1,4,16,20,31,32,33,40,63,64,65,130,1024]:
        x=rng.uniform(-20,20,n).astype(np.float32)
        x[0]=np.float32(-0.0)
        y=torch.sigmoid(torch.from_numpy(x)).numpy()
        cases.append(dict(name=f"sigmoid-body-tail-{n}",op=1,inputs=x.tolist(),expected_bits=y.view(np.uint32).tolist()))
        y=torch.nn.functional.silu(torch.from_numpy(x)).numpy()
        cases.append(dict(name=f"silu-body-tail-{n}",op=2,inputs=x.tolist(),expected_bits=y.view(np.uint32).tolist()))
        y=torch.nn.functional.softplus(torch.from_numpy(x)).numpy()
        # Windows CRT log1pf is not always correctly rounded: e.g. at
        # 0.001293487148359418 it differs from rounded float64 log1p by 1 ULP.
        # Keep that CPU comparison explicit; the independent scalar log1p
        # check below is exact, and the full-model logit gate is unchanged.
        cases.append(dict(name=f"softplus-body-tail-{n}",op=3,max_ulps=1,inputs=x.tolist(),expected_bits=y.view(np.uint32).tolist()))
    x=np.exp(rng.uniform(-20,20,4096)).astype(np.float32)
    y=np.array([math.log1p(float(v)) for v in x],dtype=np.float32)
    cases.append(dict(name="scalar-log1p-positive-domain",op=4,inputs=x.tolist(),expected_bits=y.view(np.uint32).tolist()))
    return dict(torch_revision=TORCH_REVISION,cpu_capability="AVX512",cases=cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixture",action="store_true")
    args=parser.parse_args()
    f=make_fixture()
    if args.write_fixture: FIXTURE.write_text(json.dumps(f,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text())!=f: raise AssertionError("unary fixture differs from independent reference")
    print(f"PASS unary oracle: {len(f['cases'])} cases, {sum(len(c['inputs']) for c in f['cases'])} values")


if __name__=="__main__": main()
