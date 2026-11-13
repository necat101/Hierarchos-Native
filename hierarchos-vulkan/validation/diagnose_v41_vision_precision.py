"""Locate native V4.1 stress drift against its immutable operation trace.

Consumes the released CPU fixture and an opt-in native stage capture. Never
rewrites a fixture, promotes the stress case, or changes a numerical limit.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cpu=json.loads(a.fixture.read_text());native=json.loads(a.native.read_text())
    oracle=cpu['oracle']
    if oracle['cpu']!='AVX2' or oracle['torch']!='2.14.1+cpu':raise ValueError('unexpected independent CPU provenance')
    reference=Path(__file__).parent/'reference/deepseek_v41_vl'/oracle['revision']
    if any(digest(reference/name)!=sha for name,sha in oracle['source_sha256'].items()):raise ValueError('released reference source changed')
    trace=cpu['steps'][0]['trace'];stages=native['steps'][0]['stages'];rows=[]
    pairs=[('embedded','vision.patch_embed.proj.output'),('blocks.0.n1','vision.blocks.0.norm1.output'),
        ('blocks.0.qkv','vision.blocks.0.attn.wqkv.output'),('blocks.0.qr','sdpa.0.q'),
        ('blocks.0.kr','sdpa.0.k'),('blocks.0.v','sdpa.0.v'),('blocks.0.a','sdpa.0.output'),
        ('blocks.0.o','vision.blocks.0.attn.wo.output'),('blocks.0.n2','vision.blocks.0.norm2.output'),
        ('blocks.0.gu','vision.blocks.0.mlp.w1.output'),('blocks.0.down','vision.blocks.0.mlp.w2.output'),
        ('blocks.0.out','vision.blocks.0.output'),('features','vision.output'),
        ('pre','aligner.w1.output'),('projected','aligner.output')]
    for name,key in pairs:
        tensor=trace[key]['value'];expected=np.asarray(tensor['values'],dtype=np.float32)
        if key.startswith('sdpa.'):
            expected=expected.reshape(tensor['shape'])
            expected=expected.transpose(0,2,1,3) if expected.ndim==4 else expected.transpose(1,0,2)
            expected=expected.reshape(-1)
        actual=np.asarray(stages[name],dtype=np.float32)
        if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('invalid native stage '+name)
        delta=actual.astype(np.float64)-expected
        rows.append(dict(stage=name,cpu_operation=key,values=len(expected),
            max_abs=float(abs(delta).max()),rms=float(np.sqrt(np.mean(delta*delta))),
            reference_max_magnitude=float(abs(expected).max()),
            different_words=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32))),
            within_strict_diagnostic=bool(abs(delta).max()<2e-7)))
    report=dict(status='DIAGNOSTIC_ONLY',requested=native['precision']['requested'],
        fixture_sha256=digest(a.fixture),native_capture_sha256=digest(a.native),
        independent_oracle=oracle,stages=rows,
        first_nonidentical_boundary=next((r for r in rows if r['different_words']),None),
        first_strict_diagnostic_failure=next((r for r in rows if not r['within_strict_diagnostic']),None),
        scope='native released ViT/2D rotary/aligner; bounded qualification and stress failures remain separate')
    # The patch projection is identical in FP32. Compare both norm results
    # against the same independent real-arithmetic value to expose reduction
    # rounding without treating an ideal norm as the immutable CPU oracle.
    x=np.asarray(stages['embedded'],dtype=np.float64).reshape(-1,cpu['config']['vision_dim'])
    w=np.asarray(cpu['parameters']['vision.blocks.0.norm1.weight']['values'],dtype=np.float64)
    ideal=(x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-6)*w).astype(np.float32).reshape(-1)
    norm_cpu=np.asarray(trace['vision.blocks.0.norm1.output']['value']['values'],dtype=np.float32)
    norm_native=np.asarray(stages['blocks.0.n1'],dtype=np.float32)
    report['operand_matched_first_norm']={}
    for name,value in [('independent_cpu',norm_cpu),('native',norm_native)]:
        delta=value.astype(np.float64)-ideal
        report['operand_matched_first_norm'][name]=dict(max_abs_vs_ideal=float(abs(delta).max()),
            max_ulp_vs_ideal=int(abs(value.view(np.int32).astype(np.int64)-ideal.view(np.int32).astype(np.int64)).max()))
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    for row in rows:print(row['stage'],row['max_abs'],row['rms'],flush=True)


if __name__=='__main__':main()
