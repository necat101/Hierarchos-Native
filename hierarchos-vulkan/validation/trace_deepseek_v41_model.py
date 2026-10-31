"""Observe unchanged V4.1 reference operations and localize native drift.

Writes an independent sidecar trace; never changes the acceptance fixture,
weights, input cases, reference arithmetic, or hard-gate policy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import types
import torch
import validate_deepseek_v41_model as oracle


class Observer:
    def __init__(self):
        self.steps=[]
        self.current={}
        self.pre_calls={}
        self.post_calls={}
        self.aux={}

    def save(self,layer,stage,value):
        self.current[(layer,stage)]=value.detach().clone()

    def attach(self,model):
        self.model=model
        namespace=type(model.layers[0].attn).forward.__globals__
        original_quant=namespace["act_quant"]
        def quant(x,*args,**kwargs):
            result=original_quant(x,*args,**kwargs)
            if x.ndim==4:self.save(self.active_layer,"attention_projection_input",x)
            return result
        namespace["act_quant"]=quant
        for i,block in enumerate(model.layers):
            block.attn.register_forward_pre_hook(lambda module,args,layer=i:setattr(self,"active_layer",layer))
            block.attn.wo_b.register_forward_pre_hook(lambda module,args,layer=i:self.save(layer,"attention_grouped",args[0]))
            original_mixes=block.hc_mixes
            original_pre=block.hc_pre
            original_post=block.hc_post

            def mixes(this,x,fn,scale,base,*,layer=i,original=original_mixes):
                stage="attn" if fn is this.hc_attn_fn else "ffn"
                self.save(layer,f"{stage}_projection",torch.nn.functional.linear(x.flatten(2).float(),fn))
                result=original(x,fn,scale,base)
                for name,value in zip(("pre","post","comb"),result):
                    self.save(layer,f"{stage}_{name}",value)
                return result

            def pre(this,x,mix,*,layer=i,original=original_pre):
                call=self.pre_calls.get(layer,0)
                self.pre_calls[layer]=call+1
                result=original(x,mix)
                if call<2:
                    self.save(layer,("attn_collapse","ffn_collapse")[call],result)
                return result

            def post(this,x,residual,post,comb,*,layer=i,original=original_post):
                call=self.post_calls.get(layer,0)
                self.post_calls[layer]=call+1
                result=original(x,residual,post,comb)
                self.save(layer,("after_attention","output")[call],result)
                return result

            block.hc_mixes=types.MethodType(mixes,block)
            block.hc_pre=types.MethodType(pre,block)
            block.hc_post=types.MethodType(post,block)
            block.register_forward_pre_hook(lambda module,args,layer=i:self.save(layer,"input_pre",args[2]))
            block.register_forward_pre_hook(lambda module,args,layer=i:self.save(layer,"engram",args[0]))
            if block.engram is not None:
                block.engram.register_forward_pre_hook(lambda module,args,layer=i:self.save(layer,"input",args[0]))
                block.engram.register_forward_pre_hook(lambda module,args,layer=i:self.aux.update({layer:args[1].clone()}))
                block.engram.wkv.register_forward_hook(lambda module,args,value,layer=i:self.save(layer,"engram_kv",value))
            else:
                block.register_forward_pre_hook(lambda module,args,layer=i:self.save(layer,"input",args[0]))
            for name,stage in [("attn_norm","attn_norm"),("ffn_norm","ffn_norm"),("attn","attention"),("ffn","moe")]:
                getattr(block,name).register_forward_hook(lambda module,args,value,layer=i,stage=stage:self.save(layer,stage,value))

    def begin(self,spec):
        self.current={}
        self.pre_calls={}
        self.post_calls={}
        self.aux={}

    def end(self,spec):
        # Window rows this call gathered from; compressed indices are offset by it.
        window=self.model.layers[0].attn.window_kv_cache.shape[1]
        offset=spec["seq"] if spec["pos"]==0 else window
        self.steps.append({"name":spec["name"],"index_offset":offset,"stages":[dict(layer=layer,stage=stage,values=value.flatten().tolist()) for (layer,stage),value in sorted(self.current.items())],"hashes":{str(layer):values.tolist() for layer,values in self.aux.items()}})


def drift(a,b):
    if a.shape!=b.shape: raise AssertionError((a.shape,b.shape))
    return (a.double()-b.double()).abs().max().item()


@torch.inference_mode()
def report(native,observer,fixture):
    cfg=fixture["config"]["text_config"]
    d,hc=cfg["hidden_size"],cfg["hc_mult"]
    for index,step in enumerate(native["steps"]):
        rows=fixture["batch_size"]*fixture["steps"][index]["seq"]
        actual={(x["layer"],x["stage"]):torch.tensor(x["values"]) for x in step["stages"]}
        expected={(x["layer"],x["stage"]):torch.tensor(x["values"]) for x in observer.steps[index]["stages"]}
        if actual.keys()!=expected.keys(): raise AssertionError((actual.keys()-expected.keys(),expected.keys()-actual.keys()))
        print(f"STEP {step['name']}")
        for i,block in enumerate(observer.model.layers):
            stages=[(stage,drift(values,expected[(i,stage)])) for (layer,stage),values in actual.items() if layer==i]
            print(f"  L{i} "+" ".join(f"{name}={worst:.3e}" for name,worst in stages if worst))
            get=lambda name:actual[(i,name)]
            if i==0:
                replay=block.attn(get("attn_norm").view(2,-1,d),fixture["steps"][index]["pos"]).flatten()
                for name,value in [("attention",replay),("attention_projection_input",observer.current[(i,"attention_projection_input")].flatten()),("attention_grouped",observer.current[(i,"attention_grouped")].flatten())]:
                    intrinsic=drift(get(name),value)
                    if intrinsic:print(f"    same-input {name}: {intrinsic:.17g}")
            if block.engram is not None:
                hashes=torch.tensor(observer.steps[index]["hashes"][str(i)])
                mask=torch.tensor(fixture["steps"][index]["mask"],dtype=torch.bool).view(2,-1)
                replay=block.engram(get("input").view(2,-1,hc,d),hashes,mask).flatten()
                intrinsic=drift(get("engram"),replay)
                if intrinsic: print(f"    same-input engram: {intrinsic:.17g}")
                intrinsic=drift(get("engram_kv"),observer.current[(i,"engram_kv")].flatten())
                if intrinsic: print(f"    same-hash engram_kv: {intrinsic:.17g}")
            # Replay elementary operations on identical native inputs. Keep
            # these separate from reference-sequence propagation differences.
            for name,sub,res in [("after_attention","attention","engram"),("output","moe","after_attention")]:
                prefix="attn" if sub=="attention" else "ffn"
                x=get(sub).view(2,-1,d)
                residual=get(res).view(2,-1,hc,d)
                post=get(prefix+"_post").view(2,-1,hc)
                comb=get(prefix+"_comb").view(2,-1,hc,hc)
                replay=type(block).hc_post(block,x,residual,post,comb).flatten()
                intrinsic=drift(get(name),replay)
                if intrinsic: print(f"    same-input {name}: {intrinsic:.17g}")
            for prefix,stream in [("attn","engram"),("ffn","after_attention")]:
                fn=getattr(block,"hc_"+prefix+"_fn")
                projection=torch.nn.functional.linear(get(stream).view(rows,hc*d),fn)
                intrinsic=drift(get(prefix+"_projection"),projection.flatten())
                if intrinsic: print(f"    same-input {prefix}_projection: {intrinsic:.17g}")
                mixes=type(block).hc_mixes(block,get(stream).view(2,-1,hc,d),getattr(block,"hc_"+prefix+"_fn"),getattr(block,"hc_"+prefix+"_scale"),getattr(block,"hc_"+prefix+"_base"))
                for name,value in zip(("pre","post","comb"),mixes):
                    intrinsic=drift(get(prefix+"_"+name),value.flatten())
                    if intrinsic: print(f"    same-input {prefix}_{name}: {intrinsic:.17g}")
            replay=block.ffn(get("ffn_norm").view(2,-1,d)).flatten()
            intrinsic=drift(get("moe"),replay)
            if intrinsic: print(f"    same-input moe: {intrinsic:.17g}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path",type=Path,required=True)
    parser.add_argument("--kernel-reference-path",type=Path,required=True)
    parser.add_argument("--native-output",type=Path,required=True)
    parser.add_argument("--trace-output",type=Path,required=True)
    args=parser.parse_args()
    observer=Observer()
    fixture=oracle.make_fixture(args.reference_path,args.kernel_reference_path,observer)
    if fixture!=json.loads(oracle.FIXTURE.read_text()):
        raise AssertionError("observational instrumentation changed the acceptance fixture")
    native=json.loads(args.native_output.read_text())
    # Reuse provenance/shape checks; drift rejection is expected during tracing.
    try: oracle.compare_native(fixture,args.native_output)
    except AssertionError as error:
        if "full-logit steps failed" not in str(error): raise
    args.trace_output.write_text(json.dumps({"revision":fixture["revision"],"steps":observer.steps},separators=(",",":")))
    report(native,observer,fixture)


if __name__=="__main__": main()
