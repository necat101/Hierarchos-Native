"""Qualify native V4.1 vision WMMA without replacing its released architecture.

The bounded independent fixture must pass the original strict FP32 policy.
Every mixed comparison uses the unchanged mixed precision max/RMS policy.
An optional larger-magnitude fixture remains a separate diagnostic, including
its strict FP32 failures; it cannot be promoted by the bounded result.
"""
import argparse,json,os,subprocess
from pathlib import Path
from qualify_precision_training import digest,metric,require,write,ROOT

JOINT='deepseek_v41::model::tests::training::native_vision::deepseek_v41_native_multimodal_joint_two_adamw_steps_match_frozen_cpu'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--stress-fixture',type=Path)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--libtest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);commands=[];checks={};reports={}
    source_names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard',
        'hierarchos-vulkan/src','hierarchos-vulkan/shaders','hierarchos-vulkan/Cargo.toml','hierarchos-vulkan/Cargo.lock'],text=True).splitlines()
    source_names += [str(Path(__file__).relative_to(ROOT)).replace('\\','/'),
        'hierarchos-vulkan/validation/prepare_v41_wmma_fixture.py','hierarchos-vulkan/validation/qualify_precision_training.py']
    sources={name:digest(ROOT/name) for name in set(source_names)}
    binaries={name:digest(path) for name,path in [('probe',a.binary),('libtest',a.libtest)]}
    for label,fixture in [('bounded',a.fixture),('stress',a.stress_fixture)]:
        if fixture is None:continue
        cpu=json.loads(fixture.read_text());report={};rows={}
        for mode in ('off','mixed'):
            output=a.output/f'{label}-{mode}.json'
            command=[str(a.binary.resolve()),str(fixture.resolve()),str(output.resolve())]
            result=subprocess.run(command,env={**os.environ,'HIERARCHOS_WMMA_MODE':mode},capture_output=True,text=True)
            (a.output/f'{label}-{mode}.log').write_text(result.stdout+result.stderr)
            commands.append(dict(command=command,mode=mode,returncode=result.returncode))
            require(result.returncode==0,f'{label}-{mode} failed')
            report[mode]=json.loads(output.read_text())
            precision=report[mode]['precision'];require(precision['requested']==mode,'wrong native precision')
            counts=precision['accumulation_dispatches'];phases=precision['phases']
            require(counts[1]==0,'native V4.1 silently executed Full FP16')
            require(counts[0]>0 and all(x>0 for x in phases) if mode=='mixed' else counts==[0,0],
                'missing native mixed forward, weight/input gradients or wrong FP32 route')
            reload=metric(report[mode]['steps'][2]['projected'],report[mode]['reload'],2e-7)
            require(reload['max_abs']==0,'native vision reload mismatch');rows[mode+'/reload']=reload
        def compare(name,native,cpu_values,mixed):
            strict=metric(native,cpu_values,2e-7 if label=='bounded' else float('inf'))
            strict.update(strict_limit=2e-7,within_strict_limit=strict['max_abs']<2e-7)
            rows[name+'/independent-FP32']=strict
            rows[name+'/mixed-vs-FP32']=metric(native,mixed,.02,.003)
        for i in range(3):
            x,y=report['off']['steps'][i],report['mixed']['steps'][i];c=cpu['steps'][i]
            compare(f'{i}/projected',x['projected'],c['projected']['values'],y['projected'])
            if i<2:
                compare(f'{i}/patch-gradient',x['patch_gradient'],c['patch_gradient']['values'],y['patch_gradient'])
                for key,ckey in [('gradients','gradients'),('parameters','updated_parameters')]:
                    require(x[key].keys()==y[key].keys()==c[ckey].keys(),'native tensor inventory mismatch')
                    for name,tensor in x[key].items():
                        compare(f'{i}/{key}/{name}',tensor['values'],c[ckey][name]['values'],y[key][name]['values'])
        checks[label]=rows
        reports[label]=dict(fixture_sha256=digest(fixture),precision=report['mixed']['precision'],
            independent_FP32_qualified=all(row['within_strict_limit'] for key,row in rows.items() if key.endswith('independent-FP32')),
            scope='bounded native ViT/aligner' if label=='bounded' else 'larger-magnitude diagnostic, not promoted')
    for mode in ('off','mixed'):
        command=[str(a.libtest.resolve()),JOINT,'--exact','--nocapture','--test-threads=1']
        result=subprocess.run(command,env={**os.environ,'HIERARCHOS_WMMA_MODE':mode},capture_output=True,text=True)
        log=result.stdout+result.stderr;(a.output/f'joint-{mode}.log').write_text(log)
        commands.append(dict(command=command,mode=mode,returncode=result.returncode))
        require(result.returncode==0 and '1 passed; 0 failed' in log,'native joint frozen FP32 test failed')
    require(all(digest(ROOT/name)==sha for name,sha in sources.items()),'source changed during execution')
    require(digest(a.binary)==binaries['probe'] and digest(a.libtest)==binaries['libtest'],'binary changed during execution')
    write(a.output/'qualification.json',dict(schema=1,status='PASS_BOUNDED_NATIVE_VISION',
        source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources,binary_sha256=binaries,commands=commands,reports=reports,checks=checks,
        limitations=['Native language LoRA and short vision operations remain FP32',
            'No native V4.1 automatic loss scaling or Full FP16 admission',
            'The wide joint image/language configuration has not yet been independently qualified']))
    print('NATIVE V4.1 MIXED: PASS_BOUNDED_NATIVE_VISION',flush=True)

if __name__=='__main__':main()
