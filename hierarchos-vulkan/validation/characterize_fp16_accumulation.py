"""Compare observed FP16 cooperative outputs with independent rounding models.

Diagnostic only: never changes a qualification limit or an expected tensor.
Input is wmma_qualify --dump-operands, including every original geometry/layout.
"""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();results=[]
    for sample in json.loads(args.samples.read_text()):
        m,n,k=(sample[key] for key in ('m','n','k'))
        a=np.asarray(sample['a'],dtype=np.float32).astype(np.float16).astype(np.float32)
        b=np.asarray(sample['b'],dtype=np.float32).astype(np.float16).astype(np.float32)
        a=a.reshape(k,m).T if sample['trans_a'] else a.reshape(m,k)
        b=b.reshape(n,k).T if sample['trans_b'] else b.reshape(k,n)
        actual=np.asarray(sample['actual'],dtype=np.float32).reshape(m,n)
        models={}
        # A dot2 with a binary16 destination rounds the exact two products plus
        # incoming half accumulator once. FP32 staging can double-round ties.
        acc=np.zeros((m,n),dtype=np.float64)
        for start in range(0,k,2):
            part=acc.copy()
            for t in range(start,min(start+2,k)):
                part+=a[:,t,None].astype(np.float64)*b[t,None,:].astype(np.float64)
            acc=part.astype(np.float16).astype(np.float64)
        d=actual.astype(np.float64)-acc
        models['exact_dot2_rne']=dict(max_abs=float(np.max(np.abs(d))),
            rms=float(np.sqrt(np.mean(d*d))),different=int(np.count_nonzero(actual!=acc)))
        for product_half in (False,True):
            for group in (1,2,4,8,16):
                for include_c in ('last','first'):
                    acc=np.zeros((m,n),dtype=np.float32)
                    for start in range(0,k,group):
                        part=acc.copy() if include_c=='first' else np.zeros_like(acc)
                        for t in range(start,min(start+group,k)):
                            product=a[:,t,None]*b[t,None,:]
                            if product_half:product=product.astype(np.float16).astype(np.float32)
                            part=part+product
                        acc=(part if include_c=='first' else acc+part).astype(np.float16).astype(np.float32)
                    d=actual.astype(np.float64)-acc
                    name=f'product_{"f16" if product_half else "f32"}_group_{group}_c_{include_c}'
                    models[name]=dict(max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))),
                        different=int(np.count_nonzero(actual.view(np.int32)!=acc.view(np.int32))))
        row={key:sample[key] for key in ('m','n','k','trans_a','trans_b')}
        row['models']=models;results.append(row)
        best=sorted(models,key=lambda key:models[key]['rms'])[:3]
        print(row['m'],row['n'],row['k'],row['trans_a'],row['trans_b'],[(key,models[key]) for key in best],flush=True)
    with args.output.open('x') as output:json.dump(dict(numpy=np.__version__,results=results),output,indent=2)

if __name__=='__main__':main()
