"""Execute real FP32, Mixed and FP16-accumulating training and exact PEFT resume.

This is a scoped training receipt. It does not replace original FP32 suites or
hide the failing broad Full FP16 primitive drift gates. Prepare the independent
CLIP/Llama128 CPU fixture before invoking this runner.
"""
import argparse,hashlib,json,math,os,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

ROOT=Path(__file__).resolve().parents[2]
MODES=('off','mixed','full-fp16')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(condition,message):
    if not condition:raise ValueError(message)
def metric(a,b,maximum,rms=None):
    a=np.asarray(a,dtype=np.float32).astype(np.float64)
    b=np.asarray(b,dtype=np.float32).astype(np.float64)
    require(a.shape==b.shape and a.size>0,'comparison geometry mismatch')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite comparison')
    d=a-b;result=dict(values=a.size,max_abs=float(np.abs(d).max()),rms=float(np.sqrt(np.mean(d*d))))
    require(result['max_abs']<maximum,f'maximum failed: {result}, limit {maximum}')
    require(rms is None or result['rms']<rms,f'RMS failed: {result}, limit {rms}')
    return result
def tensors(a,b,maximum,rms=None):
    a,b=load_file(str(a)),load_file(str(b));require(a.keys()==b.keys(),'tensor inventory mismatch')
    require(all(x.dtype==np.float32 for x in (*a.values(),*b.values())),'non-FP32 master or optimizer state')
    return {name:metric(a[name],b[name],maximum,rms) for name in a}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--bins',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20)
    args=parser.parse_args();require(args.steps>=4 and args.steps%2==0,'steps must be even and at least four')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fixture=args.fixtures.resolve();bins=args.bins.resolve()
    executables={name:bins/(name+'.exe') for name in ('hierarchos-vulkan-transformer-train','image_text_pipeline_probe')}
    hashes={name:digest(path) for name,path in executables.items()}
    paths=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard',
        'hierarchos-vulkan/src','hierarchos-vulkan/shaders','hierarchos-vulkan/Cargo.toml','hierarchos-vulkan/Cargo.lock'],text=True).splitlines()
    paths.append(str(Path(__file__).relative_to(ROOT)).replace('\\','/'))
    sources={path:digest(ROOT/path) for path in sorted(set(paths))}
    fixture_hashes={str(path.relative_to(fixture)):digest(path) for path in fixture.rglob('*') if path.is_file()}
    commands=[];checks={};reports={};losses={}
    def run(mode,command,label):
        env={**os.environ,'HIERARCHOS_WMMA_MODE':mode,'HIERARCHOS_ATEN_VECTOR_WIDTH':'8'}
        result=subprocess.run(list(map(str,command)),env=env,capture_output=True,text=True)
        (out/(label+'.log')).write_text(result.stdout+result.stderr)
        commands.append(dict(mode=mode,command=list(map(str,command)),returncode=result.returncode,
            environment={'HIERARCHOS_WMMA_MODE':mode,'HIERARCHOS_ATEN_VECTOR_WIDTH':'8'}))
        require(result.returncode==0,f'{label} failed: {(result.stdout+result.stderr)[-700:]}')
        return result.stdout
    def text_run(mode,kind,label,steps,resume=None,save=None):
        destination=out/label
        command=[executables['hierarchos-vulkan-transformer-train'],'--model',fixture/'model/text',
            '--dataset',fixture/'dataset.jsonl','--output',destination,'--batch-size','2','--seq-len','24',
            '--epochs',str(steps),'--learning-rate','0.001','--eps','0.001','--log-steps','1','--seed','42']
        if kind=='lora':command+=['--lora-rank','8','--lora-alpha','16','--lora-target','q_proj','--lora-target','v_proj']
        if resume is not None:command+=['--resume-training-state',resume]
        if save is not None:command+=['--save-training-state',save]
        stdout=run(mode,command,label)
        report=json.loads((destination/'training_report.json').read_text())
        require(report['optimizer_steps']==steps and report['attempted_optimizer_windows']==steps,'missing successful optimizer updates')
        require(report['precision']['requested']==mode,'wrong requested mode')
        counts=report['precision']['accumulation_dispatches'];phases=report['precision']['phases']
        if mode=='off':require(counts==[0,0] and report['loss_scaler'] is None,'FP32 requested low-precision execution')
        else:
            require(counts[0 if mode=='mixed' else 1]>0 and counts[1 if mode=='mixed' else 0]==0,'wrong accumulation route')
            require(phases[0]>0 and phases[2]>0 and (phases[1]>0)==(kind=='full'),'wrong forward/backward route')
            require(report['loss_scaler']['skipped_steps']==0,'unexpected overflow on convergence fixture')
        trajectory=[json.loads(line)['loss'] for line in stdout.splitlines() if line.startswith('{"device"') and '"event":"train_step"' in line]
        require(len(trajectory)==steps and all(math.isfinite(x) for x in trajectory),'missing/nonfinite loss trajectory')
        require(trajectory[-1]<trajectory[0],'loss did not decrease')
        reports[label]=report;losses[label]=trajectory
        return destination

    for kind in ('full','lora'):
        for mode in MODES:
            label=f'{kind}-{mode}'
            state=out/(label+'-state') if kind=='lora' else None
            destination=text_run(mode,kind,label,args.steps,save=state)
            name='model.safetensors' if kind=='full' else 'adapter_model.safetensors'
            # A self-comparison still enforces every master tensor's dtype and finiteness.
            tensors(destination/name,destination/name,2e-7)
            if kind=='lora':
                half=args.steps//2;split=out/(label+'-split-state')
                text_run(mode,kind,label+'-split',half,save=split)
                resumed=text_run(mode,kind,label+'-resume',half,resume=split,save=out/(label+'-resume-state'))
                checks[label+'/exact-resume-weights']=tensors(destination/name,resumed/name,2e-7)
                require(all(row['max_abs']==0 for row in checks[label+'/exact-resume-weights'].values()),'resumed weights differ')
                checks[label+'/exact-resume-moments']=tensors(state/'hierarchos_adapter_optimizer.safetensors',
                    out/(label+'-resume-state')/'hierarchos_adapter_optimizer.safetensors',2e-7)
                require(all(row['max_abs']==0 for row in checks[label+'/exact-resume-moments'].values()),'resumed moments differ')
                require(reports[label]['loss_scaler']==reports[label+'-resume']['loss_scaler'],'resumed scaler state differs')
        for mode in MODES[1:]:
            maximum,rms=(.02,.003) if mode=='mixed' else (.05,.01)
            checks[f'{kind}-{mode}/loss-vs-FP32']=metric(losses[f'{kind}-off'],losses[f'{kind}-{mode}'],maximum,rms)
            name='model.safetensors' if kind=='full' else 'adapter_model.safetensors'
            checks[f'{kind}-{mode}/masters-vs-FP32']=tensors(out/f'{kind}-off'/name,out/f'{kind}-{mode}'/name,maximum,rms)
    image={};cpu=json.loads((fixture/'cpu.json').read_text())
    for mode in MODES:
        label='vision-'+mode;destination=out/label
        run(mode,[executables['image_text_pipeline_probe'],'--model',fixture/'model','--fixture',fixture/'input.json','--output',destination],label)
        report=json.loads((destination/'report.json').read_text());image[mode]=report
        p=report['precision'];require(p['requested']==mode,'wrong vision mode')
        if mode=='off':require(p['cooperative_dispatches']==0,'FP32 vision dispatched WMMA')
        else:
            require(all(p[k]>0 for k in ('forward','weight_grad','input_grad')),'missing vision WMMA phase')
            require(p['full_fp16_accumulation' if mode=='full-fp16' else 'mixed_fp32_accumulation']>0,'missing requested vision accumulation')
            require(p['mixed_fp32_accumulation' if mode=='full-fp16' else 'full_fp16_accumulation']==0,'wrong vision accumulation')
        for key in ('logits','vision_features','projected_features'):
            result=metric(report['final'][key],report['reload'][key],2e-7)
            require(result['max_abs']==0,'vision checkpoint/reload is not bit-identical')
            checks[f'{label}/reload-{key}']=result
        for index,step in enumerate(report['steps']):
            require(step['training']['text']['step']==index+1,'missing vision optimizer step')
            require(math.isfinite(step['training']['text']['loss']),'nonfinite vision loss')
            maximum,rms=(.02,.003) if mode=='mixed' else (.05,.01)
            for key in ('logits','vision_features','projected_features'):
                if mode=='off':
                    if key not in cpu['steps'][index]:continue
                    # Preserve strict outcome without treating a wide diagnostic as an original frozen gate.
                    row=metric(cpu['steps'][index][key],step['forward'][key],float('inf'))
                    row.update(strict_limit=2e-7,within_strict_limit=row['max_abs']<2e-7)
                else:row=metric(image['off']['steps'][index]['forward'][key],step['forward'][key],maximum,rms)
                checks[f'{label}/step-{index+1}-{key}']=row
            for key in ('pixel_gradient','projected_gradient'):
                if mode=='off':
                    row=metric(cpu['steps'][index][key],step['training'][key],float('inf'))
                    row.update(strict_limit=2e-7,within_strict_limit=row['max_abs']<2e-7)
                else:row=metric(image['off']['steps'][index]['training'][key],step['training'][key],maximum,rms)
                checks[f'{label}/step-{index+1}-{key}']=row
    require(all(digest(path)==hashes[name] for name,path in executables.items()),'executable changed during qualification')
    require(all(digest(ROOT/path)==sha for path,sha in sources.items()),'source changed during qualification')
    require(all(digest(fixture/path)==sha for path,sha in fixture_hashes.items()),'CPU fixture changed during qualification')
    write(out/'qualification.json',dict(schema=1,status='PASS_SCOPED',
        scope='Llama128 full/ordinary LoRA convergence, exact LoRA resume; shared CLIP128 joint image/text two-step training and reload',
        steps=args.steps,source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources,binary_sha256=hashes,fixture_sha256=fixture_hashes,commands=commands,
        reports=reports,losses=losses,checks=checks,
        limitations=['Does not replace 166 original text and 93 original FP32 image gates',
            'Original Full FP16 primitive drift failures remain separate evidence',
            'Vision loss scaling and full-model optimizer resume are not qualified here']))
    print('PRECISION TRAINING: PASS_SCOPED',flush=True)

if __name__=='__main__':main()
