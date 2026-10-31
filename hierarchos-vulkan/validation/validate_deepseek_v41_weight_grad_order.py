"""Pin CPU dW GEMM reduction from its output geometry, including short K tails."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
FIXTURE=HERE/'fixtures/deepseek_v41_weight_grad_order.json'
def bits(x):return x.contiguous().view(torch.uint32).flatten().tolist()
def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(9187)
    cases=[]
    for rows in [1,2,3,5,10,16,33]:
        for inp,out in [(32,1),(32,2),(32,3),(2,32),(4,32),(8,32),(6,3),(16,4)]:
            x=torch.from_numpy(rng.normal(0,.3,(rows,inp)).astype(np.float32))
            g=torch.from_numpy(rng.normal(0,.01,(rows,out)).astype(np.float32))
            expected=g.T@x
            cases.append(dict(rows=rows,input_dim=inp,output_dim=out,x_bits=bits(x),grad_bits=bits(g),weight_grad_bits=bits(expected)))
    payload=dict(torch_version=torch.__version__,torch_git_version=torch.version.git_version,seed=9187,cases=cases)
    if FIXTURE.exists() and json.loads(FIXTURE.read_text())!=payload:raise SystemExit('weight-gradient CPU fixture changed; refusing overwrite')
    FIXTURE.write_text(json.dumps(payload,indent=2)+'\n')
    print('PASS weight-gradient CPU fixture',len(cases),'cases',hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
if __name__=='__main__':main()
