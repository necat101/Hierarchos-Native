"""Generate analytic log center data and freeze independent Windows CRT outputs.

The 129 centers are 1+i/128. Log heads discard twelve mantissa bits; tails
truncate toward zero; reciprocals round to FP32. No DLL bytes, model inputs or
native results are used to construct the numeric table. Expected function
values come from the CPU's scalar CRT on separate deterministic inputs.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/"shaders/fp32_v41_cpu_log_table.glsl"
FIXTURE=ROOT/"validation/fixtures/deepseek_v41_cpu_log.json"


def analytic_table():
    rows=[]
    for i in range(129):
        value=math.log1p(i/128)
        head=np.float32(value).view(np.uint32)&np.uint32(0xfffff000)
        residual=value-float(head.view(np.float32))
        tail=np.float32(residual)
        if float(tail)>residual:
            tail=np.nextafter(tail,np.float32(0))
        inverse=np.float32(256/(128+i))
        rows.append((int(head),int(tail.view(np.uint32)),int(inverse.view(np.uint32))))
    body=",\n".join("    uvec3("+",".join(f"0x{v:08x}u" for v in row)+")" for row in rows)
    return "// Generated analytic log(center) high/low and reciprocal; see validation/validate_deepseek_v41_cpu_log.py.\nconst uvec3 v41_cpu_log_centers[129]=uvec3[129](\n"+body+"\n);\n"


def fixture(table):
    rng=np.random.default_rng(4141108)
    inputs=np.exp(rng.uniform(-20,20,4096)).astype(np.float32)
    centers=np.float32(1)+np.arange(129,dtype=np.float32)/np.float32(128)
    inputs=np.concatenate((inputs,np.nextafter(centers,np.float32(0)),centers,
                           np.nextafter(centers,np.float32(np.inf))))
    crt=ctypes.CDLL("ucrtbase.dll")
    result=dict(policy="windows_crt_scalar_log_fma_v1",table_sha256=hashlib.sha256(table.encode()).hexdigest(),
                input_bits=inputs.view(np.uint32).tolist())
    for name in ("logf","log1pf"):
        function=getattr(crt,name)
        function.argtypes=[ctypes.c_float]
        function.restype=ctypes.c_float
        result[name+"_bits"]=np.array([function(float(v)) for v in inputs],np.float32).view(np.uint32).tolist()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-fixture",action="store_true")
    args=p.parse_args()
    if args.write_fixture and (TABLE.exists() or FIXTURE.exists()):
        raise FileExistsError("scalar log data is frozen; repair native math")
    table=analytic_table()
    result=fixture(table)
    if args.write_fixture:
        TABLE.write_text(table,newline="\n")
        FIXTURE.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    elif TABLE.read_text()!=table or json.loads(FIXTURE.read_text())!=result:
        raise AssertionError("analytic table or scalar CPU reference changed")
    print("PASS analytic CPU log table and",2*len(result["input_bits"]),"scalar reference values")


if __name__=="__main__":
    main()
