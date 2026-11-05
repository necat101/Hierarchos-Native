"""Freeze identical-gradient PyTorch AdamW states, then replay only on Vulkan."""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
from qualify_multimodal_training import digest,write_json
from qualify_llava_pipeline import oracle_env

def prepare(args):
    import torch
    torch.set_num_threads(1)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest={"complete":False,"torch":torch.__version__,"torch_revision":torch.version.git_version,
        "cpu":torch.backends.cpu.get_cpu_capability(),"threads":1,"cases":[]}
    for index,(length,beta1,beta2) in enumerate([(19,.9,.999),(32,.5,.95),(65,.1,.999)]):
        torch.manual_seed(0xAD400+index)
        initial=torch.randn(length)*.02
        gradients=[torch.randn(length)*.002 for _ in range(3)]
        hyper={"lr":.001,"beta1":beta1,"beta2":beta2,"eps":.001,"weight_decay":.01}
        p=torch.nn.Parameter(initial.clone())
        optimizer=torch.optim.AdamW([p],lr=hyper["lr"],betas=(beta1,beta2),eps=hyper["eps"],weight_decay=hyper["weight_decay"],foreach=False,fused=False)
        steps=[]
        for gradient in gradients:
            p.grad=gradient;optimizer.step();state=optimizer.state[p]
            steps.append({"parameter":p.detach().tolist(),"moment":state["exp_avg"].tolist(),"square":state["exp_avg_sq"].tolist()})
        name=f"len-{length}"
        write_json(out/f"{name}.json",{"initial":initial.tolist(),"gradients":[g.tolist() for g in gradients],"hyper":hyper,"decay":True})
        write_json(out/f"{name}-cpu.json",{"steps":steps})
        manifest["cases"].append({"name":name,"input_sha256":digest(out/f"{name}.json"),"cpu_sha256":digest(out/f"{name}-cpu.json")})
    manifest["complete"]=True;write_json(out/"manifest.json",manifest)

def run(args):
    root=args.fixtures.resolve();manifest=json.loads((root/"manifest.json").read_text())
    if not manifest["complete"] or not manifest.get("cases"):raise ValueError("incomplete or empty optimizer reference")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    results=[]
    for case in manifest["cases"]:
        name=case["name"];source=root/f"{name}.json";cpu=root/f"{name}-cpu.json"
        if digest(source)!=case["input_sha256"] or digest(cpu)!=case["cpu_sha256"]:raise ValueError("frozen optimizer state changed")
        p=subprocess.run([str(args.binary.resolve()),str(source)],capture_output=True,text=True,check=True,env=oracle_env(manifest["cpu"]))
        actual=json.loads(p.stdout);write_json(out/f"{name}.json",actual)
        expected=json.loads(cpu.read_text());metrics=[]
        for index,(a,b) in enumerate(zip(actual["steps"],expected["steps"],strict=True),1):
            for key in ["parameter","moment","square"]:
                av,bv=a[key],b[key]
                if not av or len(av)!=len(bv) or not all(math.isfinite(v) for v in av+bv):raise ValueError("invalid optimizer comparison coverage")
                differences=sum(struct.pack('<f',x)!=struct.pack('<f',y) for x,y in zip(av,bv))
                metrics.append({"step":index,"tensor":key,"values":len(av),"word_differences":differences,
                    "max_abs":max(abs(x-y) for x,y in zip(av,bv)),"pass":differences==0})
        passed=all(m["pass"] for m in metrics)
        results.append({"case":name,"pass":passed,"metrics":metrics})
        print(name,passed,[m for m in metrics if not m['pass']],flush=True)
    write_json(out/"qualification.json",{"scope":"identical_gradients_cpu_adamw","bit_exact":True,
        "pass":all(r["pass"] for r in results),"results":results,"binary_sha256":digest(args.binary),"runner_sha256":digest(__file__)})
    if not all(r["pass"] for r in results):raise SystemExit(1)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("prepare");a.add_argument("--output",type=Path,required=True)
    a=s.add_parser("run");a.add_argument("--output",type=Path,required=True);a.add_argument("--fixtures",type=Path,required=True);a.add_argument("--binary",type=Path,required=True)
    args=p.parse_args();(prepare if args.command=="prepare" else run)(args)
