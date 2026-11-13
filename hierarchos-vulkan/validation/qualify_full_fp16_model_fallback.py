"""Exercise internal oversized fallback through joint training and HF export.

This is a fallback diagnostic on the additional 128-wide package, not Full
FP16 admission for that package. The frontend must continue to reject it. The
separate K=128 component test proves exact forward/dW/dX FP32 fallback and AdamW
moment restoration; this run additionally exercises the real package exporter.
"""
import argparse, json, os, subprocess
from pathlib import Path
from qualify_precision_training import ROOT, digest, metric, require, tensors, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();fixture=a.fixture.resolve();binary=a.binary.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads((fixture/'model/text/config.json').read_text())
    require(config['hidden_size']==128,'this fallback diagnostic requires the retained 128-wide package')
    names=['input.json',*[f.relative_to(fixture).as_posix() for f in (fixture/'model').rglob('*') if f.is_file()]]
    inputs={name:digest(fixture/name) for name in names};binary_sha=digest(binary)
    reports={};commands=[]
    for mode in ('off','full-fp16'):
        command=[str(binary),'--model',str(fixture/'model'),'--fixture',str(fixture/'input.json'),'--output',str(out/mode)]
        env={**os.environ,'HIERARCHOS_WMMA_MODE':mode,'HIERARCHOS_ATEN_VECTOR_WIDTH':'8'}
        r=subprocess.run(command,env=env,cwd=ROOT,capture_output=True,text=True)
        (out/(mode+'.log')).write_text(r.stdout+r.stderr)
        commands.append(dict(command=command,mode=mode,exit_code=r.returncode))
        require(r.returncode==0,'fallback package execution failed: '+mode)
        reports[mode]=json.loads((out/mode/'report.json').read_text())
        require([x['training']['text']['step'] for x in reports[mode]['steps']]==[1,2],'missing joint updates')
        require(reports[mode]['precision']['master_weights']=='fp32' and
                reports[mode]['precision']['optimizer_state']=='fp32','fallback changed master or moment storage')
    precision=reports['full-fp16']['precision']
    require(precision['full_fp16_reduction_fallbacks']>0,'oversized model never executed FP32 fallback')
    require(precision['mixed_fp32_accumulation']==0,'Full fallback silently used Mixed accumulation')
    checks={}
    for mode in reports:
        for tensor in ('logits','vision_features','projected_features'):
            row=metric(reports[mode]['final'][tensor],reports[mode]['reload'][tensor],2e-7)
            require(row['max_abs']==0,'export/reload changed fallback output')
            checks[mode+'/reload/'+tensor]=row
    for tensor in ('logits','vision_features','projected_features'):
        checks['full-vs-fp32/'+tensor]=metric(reports['full-fp16']['final'][tensor],reports['off']['final'][tensor],.05,.01)
    for component in ('image','text'):
        checks['masters/'+component]=tensors(out/'full-fp16/step-2'/component/'model.safetensors',
                                             out/'off/step-2'/component/'model.safetensors',.05,.01)
    require(digest(binary)==binary_sha and all(digest(fixture/name)==sha for name,sha in inputs.items()),'fallback inputs or executable changed')
    write(out/'qualification.json',dict(schema=1,status='PASS_INTERNAL_FALLBACK_ONLY',
        scope='Two joint updates, oversized FP32 fallback, FP32 masters/moments and native HF export/reload; no 128-wide Full FP16 frontend admission',
        model_config=config,fixture_sha256=inputs,binary_sha256=binary_sha,commands=commands,
        reports=reports,checks=checks,source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=digest(__file__),limits=dict(full_max_abs=.05,full_rms=.01,reload_exact=True)))
    print('INTERNAL FULL FP16 MODEL FALLBACK: PASS; 128-wide frontend admission remains rejected',flush=True)


if __name__=='__main__':main()
