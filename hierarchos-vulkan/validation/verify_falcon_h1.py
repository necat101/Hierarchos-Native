#!/usr/bin/env python3
"""Strict Falcon H1 validation against the local Transformers checkout (never production)."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from verify_hf_logits import ROOT, REPO, save_reference_checkpoint, transformers

ATOL = 2e-7

def native_generation(binary, model_dir, fixture_path, use_cache, device_index):
    """Fail if a cache-labelled run silently falls back to full-prefix math."""
    output = model_dir / ("native-cache.json" if use_cache else "native-full.json")
    command = [str(binary), "--model", str(model_dir), "--fixture", str(fixture_path),
               "--output", str(output), "--use-cache", str(use_cache).lower()]
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "HIERARCHOS_FALCON_H1_TRACE_CACHE": "1"})
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    count = proc.stderr.count("Falcon H1 native recurrent dispatch: batch=1, seq=1")
    config = json.loads((model_dir / "config.json").read_text())
    fixture = json.loads(fixture_path.read_text())
    expected = config["num_hidden_layers"] * (fixture["max_new_tokens"] - 1) if use_cache else 0
    assert count == expected, f"expected {expected} recurrent dispatches, got {count}"
    result = json.loads(output.read_text())
    result["recurrent_dispatches"] = count
    return result

def dual_adamw(model,folder):
    """Second AdamW trajectory uses a small epsilon, with every step checked."""
    from safetensors.torch import load_file
    folder.mkdir(parents=True,exist_ok=True);source=folder/"source"
    save_reference_checkpoint("falcon_h1",model,source)
    model.train()
    ids=torch.tensor([[1,5,9,3,7,2,6]])
    mask=torch.tensor([[1,1,0,1,1,1,1]])
    targets=torch.tensor([[5,9,3,7,2,6,-100]])
    hyper={"learning_rate":0.0002,"beta1":0.85,"beta2":0.97,"eps":1e-8,"weight_decay":0.0}
    fixture={"batch_size":1,"seq_len":7,"input_ids":ids.flatten().tolist(),
        "targets":[int(x) if x>=0 else 0xFFFFFFFF for x in targets.flatten()],
        "attention_mask":mask.flatten().float().tolist(),"loss_weights":targets.ne(-100).float().flatten().tolist(),**hyper}
    fp=folder/"training.json";fp.write_text(json.dumps(fixture))
    optim=torch.optim.AdamW(model.parameters(),lr=hyper["learning_rate"],betas=(hyper["beta1"],hyper["beta2"]),eps=hyper["eps"],weight_decay=0,foreach=False)
    results=[]
    for step in [1,2]:
        destination=folder/f"step{step}"
        proc=subprocess.run([str(ROOT/"target/debug/transformer_parity.exe"),"--model",str(source),"--fixture",str(fp),"--output",str(destination),"--steps",str(step)],capture_output=True,text=True,cwd=REPO)
        if proc.returncode: raise RuntimeError(proc.stderr)
        optim.zero_grad(set_to_none=True)
        logits=model(ids,attention_mask=mask,use_cache=False).logits
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten());loss.backward();optim.step()
        saved=load_file(destination/"model.safetensors")
        for n,p in model.named_parameters():
            if not torch.isfinite(saved[n]).all() or not torch.isfinite(p).all():
                raise AssertionError(f"dual AdamW {folder.name} step {step}: nonfinite {n}")
        drifts=sorted(((float((saved[n]-p.detach()).abs().max()),n) for n,p in model.named_parameters()),reverse=True)
        (folder/f"step{step}-drift.json").write_text(json.dumps(drifts,indent=2))
        print(f"dual AdamW {folder.name} step {step}: {drifts[:5]}",flush=True)
        results.append({"step":step,"max_abs":drifts[0][0],"worst":drifts[0][1]})
    (folder/"dual-report.json").write_text(json.dumps(results,indent=2))
    if any(r["max_abs"]>ATOL for r in results):
        raise AssertionError(f"dual AdamW {folder.name}: {results}")
    return results

def roundtrip(folder,source):
    """Both runtimes load exactly the same trained native checkpoint."""
    from verify_smollm3_generation import hf_generation
    loaded=transformers.FalconH1ForCausalLM.from_pretrained(source,attn_implementation="eager").eval()
    ids=torch.tensor([[1,5,9,3,7,2,6]])
    mask=torch.tensor([[1,1,1,0,1,1,1]])
    fp=folder/"reload-input.json";fp.write_text(json.dumps({"batch_size":1,"seq_len":7,"input_ids":ids.flatten().tolist(),"attention_mask":mask.flatten().float().tolist()}))
    out=folder/"reload-logits.json";export=folder/"reexport"
    subprocess.run([str(ROOT/"target/debug/hierarchos-vulkan-transformer-logits.exe"),"--model",str(source),"--fixture",str(fp),"--output",str(out),"--export-model",str(export)],check=True,cwd=REPO)
    with torch.no_grad():reference=loaded(ids,attention_mask=mask,use_cache=False).logits.flatten()
    result={"logits":compare(torch.tensor(json.loads(out.read_text())["logits"]),reference,"trained reload")}
    from safetensors.torch import load_file
    original=load_file(source/"model.safetensors");saved=load_file(export/"model.safetensors")
    assert original.keys()==saved.keys()
    assert all(torch.equal(v,saved[k]) for k,v in original.items()),"re-export changed parameter values"
    fixture=folder/"reload-generation.json";fixture.write_text(json.dumps({"input_ids":[1,5,9],"max_new_tokens":5}))
    sequence,reference=hf_generation(loaded,torch.tensor([[1,5,9]]),True,5)
    native=native_generation(ROOT/"target/debug/transformer_generation_parity.exe",export,fixture,True,None)
    assert sequence==native["sequence"]
    result["generation"]=compare_steps([torch.tensor(x) for x in native["logits"]],reference,"trained reexport generation")
    return result

def trace_failure(model,ids,mask,command,folder):
    from transformers.models.falcon_h1 import modeling_falcon_h1 as oracle
    from unittest.mock import patch
    trace={};hooks=[];active=[0]
    def put(name,x): trace[name]=x.detach().float().reshape(-1)
    def output(name,scale=1.0):
        def hook(_module,_input,out):put(name,(out[0] if isinstance(out,tuple) else out)*scale)
        return hook
    hooks.append(model.model.embed_tokens.register_forward_hook(output("embed",model.config.embedding_multiplier)))
    for i,layer in enumerate(model.model.layers):
        prefix=f"layers.{i}"
        def select(_module,_input,idx=i):active[0]=idx
        hooks.append(layer.register_forward_pre_hook(select))
        for module,name,scale in [
            (layer.input_layernorm,"ln1",1),(layer.mamba.in_proj,"mamba.in_proj",1),
            (layer.mamba,"mamba.output",model.config.ssm_out_multiplier),
            (layer.self_attn.q_proj,"q",1),(layer.self_attn.k_proj,"k",model.config.key_multiplier),
            (layer.self_attn.v_proj,"v",1),(layer.self_attn.o_proj,"attention.output",1),
            (layer.pre_ff_layernorm,"ln2",1),(layer.feed_forward.gate_proj,"mlp.gate",model.config.mlp_multipliers[0]),
            (layer.feed_forward,"mlp.output",1),(layer,"output",1)]:
            hooks.append(module.register_forward_hook(output(prefix+"."+name,scale)))
        def inp(_module,args,name=prefix+".mamba.gated"):put(name,args[0])
        hooks.append(layer.mamba.out_proj.register_forward_pre_hook(inp))
        def residual(_module,args,name=prefix+".residual"):put(name,args[0])
        hooks.append(layer.pre_ff_layernorm.register_forward_pre_hook(residual))
    original=oracle.mamba2_chunk_scan
    def scan(*args,**kwargs):
        x,dt,A,B,C=args[:5];prefix=f"layers.{active[0]}"
        put(prefix+".mamba.conv",torch.cat([x.flatten(2),B.flatten(2),C.flatten(2)],-1))
        out=original(*args,**kwargs);put(prefix+".mamba.scan",out[0] if isinstance(out,tuple) else out)
        return out
    with torch.no_grad(),patch.object(oracle,"mamba2_chunk_scan",scan):model(ids,attention_mask=mask,use_cache=False)
    for h in hooks:h.remove()
    path=folder/"native-trace.json"
    subprocess.run(command,check=True,cwd=REPO,env={**os.environ,"HIERARCHOS_FALCON_H1_TRACE":str(path)})
    native=json.loads(path.read_text());deltas={}
    for name in trace:
        if name in native:
            deltas[name]=(torch.tensor(native[name])-trace[name]).abs().max().item()
    (folder/"trace-drift.json").write_text(json.dumps(deltas,indent=2))
    print(json.dumps(deltas,indent=2),flush=True)

def tiny(norm=True, before=False, groups=2, biases=False):
    torch.manual_seed(240921)
    c = transformers.FalconH1Config(
        vocab_size=41, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=64, mamba_d_ssm=24, mamba_n_heads=4,
        mamba_d_head=6, mamba_n_groups=groups, mamba_d_state=4, mamba_d_conv=3,
        mamba_chunk_size=4, mamba_rms_norm=norm, mamba_norm_before_gate=before,
        attention_bias=biases, mlp_bias=biases, mamba_proj_bias=biases, projectors_bias=biases,
        attention_dropout=0.0, embedding_multiplier=1.7, lm_head_multiplier=0.7,
        attention_in_multiplier=0.8, attention_out_multiplier=0.6, key_multiplier=0.3,
        ssm_in_multiplier=0.9, ssm_out_multiplier=0.8,
        ssm_multipliers=[0.7,0.8,0.6,0.9,0.5], mlp_multipliers=[0.8,0.7],
        rope_parameters={"rope_type":"default", "rope_theta":100000.0},
    )
    c._attn_implementation="eager"
    return transformers.FalconH1ForCausalLM(c).eval()

def compare(a,b,label):
    if a.shape != b.shape: raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise AssertionError(f"{label}: nonfinite")
    drift=(a-b).abs().max().item()
    print(f"{label}: {drift:.10g}",flush=True)
    if drift>ATOL: raise AssertionError(f"{label}: {drift} > {ATOL}")
    return drift

def compare_steps(lhs,rhs,label):
    if len(lhs)!=len(rhs): raise AssertionError(f"{label}: step count mismatch")
    per_step=[compare(a,b,f"{label}/token{i}") for i,(a,b) in enumerate(zip(lhs,rhs,strict=True))]
    return {"max_abs":max(per_step,default=0.0),"per_step_max_abs":per_step}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT/".falcon-h1-fixtures")
    p.add_argument("--training",action="store_true")
    p.add_argument("--generation",action="store_true")
    p.add_argument("--keep-going",action="store_true",help="Record failed training phases and run remaining checks; still exits nonzero on failure")
    p.add_argument("--variants",nargs="+",choices=["gated_after","gated_before","silu"],default=["gated_after","gated_before","silu"])
    args=p.parse_args()
    torch.set_num_threads(1)
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    report={"atol":ATOL,"oracle":transformers.__file__,"fixtures":{},"passed":False}
    # An interrupted/failed rerun must not leave a stale successful report.
    (root/"report.json").write_text(json.dumps(report,indent=2))
    failures=[]
    for name,norm,before,groups,bias in [("gated_after",True,False,2,False),("gated_before",True,True,2,True),("silu",False,False,1,False)]:
        if name not in args.variants: continue
        model=tiny(norm,before,groups,bias);folder=root/name
        save_reference_checkpoint("falcon_h1",model,folder)
        result={}
        for seq in [1,3,4,7,9]:
            ids=torch.tensor([[1,5,9,3,7,2,6,8,4][:seq]])
            for padded in [False,True] if seq>1 else [False]:
                mask=torch.ones_like(ids)
                if padded: mask[0,0]=0;mask[0,-1]=0
                fixture={"batch_size":1,"seq_len":seq,"input_ids":ids.flatten().tolist(),"attention_mask":mask.flatten().float().tolist()}
                fp=folder/"input.json";fp.write_text(json.dumps(fixture))
                output=folder/"logits.json"
                command=[str(ROOT/"target/debug/hierarchos-vulkan-transformer-logits.exe"),"--model",str(folder),"--fixture",str(fp),"--output",str(output)]
                subprocess.run(command,cwd=REPO,check=True)
                with torch.no_grad(): reference=model(ids,attention_mask=mask,use_cache=False).logits
                native=torch.tensor(json.loads(output.read_text())["logits"]).reshape_as(reference)
                label=f"{name}/seq{seq}/pad{padded}"
                try: result[label]=compare(native,reference,label)
                except AssertionError:
                    trace_failure(model,ids,mask,command,folder)
                    raise
        report["fixtures"][name]=result
        if args.generation:
            from verify_smollm3_generation import hf_generation
            fixture=folder/"generation.json"
            fixture.write_text(json.dumps({"input_ids":[1,5,9],"max_new_tokens":5}))
            ids=torch.tensor([[1,5,9]])
            hf_seq,hf_logits=hf_generation(model,ids,True,5)
            full_seq,full_logits=hf_generation(model,ids,False,5)
            native_cache=native_generation(ROOT/"target/debug/transformer_generation_parity.exe",folder,fixture,True,None)
            native_full=native_generation(ROOT/"target/debug/transformer_generation_parity.exe",folder,fixture,False,None)
            assert hf_seq==full_seq==native_cache["sequence"]==native_full["sequence"]
            cached=[torch.tensor(x) for x in native_cache["logits"]]
            full=[torch.tensor(x) for x in native_full["logits"]]
            result["cache_hf"]=compare_steps(cached,hf_logits,name+" cache/HF")
            result["full_hf"]=compare_steps(full,full_logits,name+" full/HF")
            result["cache_full"]=compare_steps(cached,full,name+" cache/full")
            result["recurrent_dispatches"]=native_cache["recurrent_dispatches"]
            print(json.dumps(result["cache_hf"]),flush=True)
            for prompt in [[1], [1,5,9,3], [1,5,9,3,7,2,6], [1,5,9,3,7,2,6,8,4]]:
                fixture.write_text(json.dumps({"input_ids":prompt,"max_new_tokens":5}))
                hf_seq,hf_logits=hf_generation(model,torch.tensor([prompt]),True,5)
                native_cache=native_generation(ROOT/"target/debug/transformer_generation_parity.exe",folder,fixture,True,None)
                native_full=native_generation(ROOT/"target/debug/transformer_generation_parity.exe",folder,fixture,False,None)
                assert hf_seq==native_cache["sequence"]==native_full["sequence"]
                cached=[torch.tensor(x) for x in native_cache["logits"]]
                full=[torch.tensor(x) for x in native_full["logits"]]
                label=f"prefill{len(prompt)}"
                result[label]={"cache_hf":compare_steps(cached,hf_logits,name+" "+label+" cache/HF"),
                               "cache_full":compare_steps(cached,full,name+" "+label+" cache/full"),
                               "recurrent_dispatches":native_cache["recurrent_dispatches"]}
        if args.training:
            from verify_hf_training import compare_training
            for phase,run in [
                ("adamw_step1",lambda: compare_training(ROOT/"target/debug/transformer_parity.exe",root/"training",name,tiny(norm,before,groups,bias),None,steps=1)),
                ("adamw",lambda: compare_training(ROOT/"target/debug/transformer_parity.exe",root/"training",name,tiny(norm,before,groups,bias),None,steps=2)),
                ("dual_adamw",lambda: dual_adamw(tiny(norm,before,groups,bias),root/"dual"/name)),
                ("roundtrip",lambda: roundtrip(folder,root/"training"/name/"trained")),
            ]:
                try:
                    result[phase]=run()
                    print(json.dumps(result[phase]),flush=True)
                except (AssertionError,RuntimeError,subprocess.CalledProcessError) as error:
                    result[phase]={"failed":str(error)}
                    failures.append(f"{name}/{phase}: {error}")
                    print(f"FAIL {failures[-1]}",flush=True)
                    if not args.keep_going:
                        (root/"report.json").write_text(json.dumps(report,indent=2))
                        raise
        (root/"report.json").write_text(json.dumps(report,indent=2))
    report["failures"]=failures
    report["passed"]=not failures
    (root/"report.json").write_text(json.dumps(report,indent=2))
    if failures: raise AssertionError("Strict Falcon H1 failures: "+"; ".join(failures))

if __name__=="__main__":main()
