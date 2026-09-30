"""Compare convolution/activation staging with the single-threaded oracle."""
import json
import numpy as np
import torch
import verify_falcon_h1 as v

torch.set_num_threads(1)
model=v.tiny(); m=model.model.layers[0].mamba
t=json.loads((v.ROOT/'.falcon-h1-fixtures/dual/gated_after/native-backward-trace.json').read_text())
proj=torch.tensor(t['forward.layers.0.mamba.in_proj']).reshape(7,-1)
x=proj[:,24:64]*torch.tensor([0.8]*24+[0.6]*8+[0.9]*8)
w=m.conv1d.weight.detach()[:,0,:]; bias=m.conv1d.bias.detach()
pre=torch.nn.functional.conv1d(x.T[None],w[:,None],bias,padding=2,groups=40)[0,:,:7].T
mask=torch.tensor([1,1,0,1,1,1,1])[:,None]
native=torch.tensor(t['forward.layers.0.mamba.conv']).reshape(7,40)
ref=torch.nn.functional.silu(pre)*mask
print('native/ref',float((native-ref).abs().max()),int((native!=ref).sum()))
for fused in [False,True]:
    a=np.broadcast_to(bias.numpy(),(7,40)).copy()
    for k in range(3):
        xx=np.zeros((7,40),np.float32)
        xx[2-k:]=x.numpy()[:5+k]
        prod=xx.astype(np.float64)*w[:,k].numpy().astype(np.float64)
        if not fused:prod=prod.astype(np.float32)
        a=(a.astype(np.float64)+prod).astype(np.float32)
    a=torch.tensor(a)
    print('pre',fused,float((a-pre).abs().max()),int((a!=pre).sum()))
    for name,y in [('silu',torch.nn.functional.silu(a)),('div',a/(1+torch.exp(-a))),('mul',a*(1/(1+torch.exp(-a))))]:
        y=y*mask
        print(fused,name,'native',float((y-native).abs().max()),int((y!=native).sum()),'HF',float((y-ref).abs().max()),int((y!=ref).sum()))
