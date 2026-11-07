"""Replay the complete green Gen9 multimodal matrix on a physical AMD adapter.

Original frozen AVX2 oracles are read-only. Native AVX512 image qualification is
a separate track. GPU jobs run serially; build before invoking this runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

CRATE=Path(__file__).resolve().parents[1]
BASELINE=CRATE/'validation/release_parity/multimodal_training/20261105-gen9-v41-upload-batching'
IMAGES={
    'shared77':('qualify_shared_image_pipeline','target/multimodal-gen9/shared-image-cpu-v3','image_text_pipeline_probe',77,'shared77-qualification.json'),
    'seq2seq':('qualify_seq2seq_image_pipeline','target/multimodal-gen9/seq2seq-image-cpu-v1','image_text_pipeline_probe',6,'seq2seq-qualification.json'),
    'gemma4-legacy':('qualify_gemma4_pipeline','hierarchos-vulkan/validation/tmp-gemma4-fixtures/gen4-cpu','gemma4_vision_probe',1,'gemma4-legacy-qualification.json'),
    'gemma4-rgb':('qualify_gemma4_pipeline','target/multimodal-gen9/gemma4-rgb-cpu','gemma4_vision_probe',1,'gemma4-rgb-qualification.json'),
    'gemma3':('qualify_gemma3_pipeline','target/multimodal-gen9/gemma3-cpu','gemma3_vision_probe',4,'gemma3-qualification.json'),
    'llava':('qualify_llava_pipeline','target/multimodal-gen9/llava-cpu','llava_pipeline_probe',4,'llava-qualification.json'),
}
STAGES=[*IMAGES,'text166','deepseek215','native-v41']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_bytes((json.dumps(value,indent=2)+'\n').encode())
def case_names(q):return {r.get('case',r.get('name')) for r in q['results']}
def check(condition,message):
    if not condition:raise ValueError(message)

def image_result(q,label):
    _,_,_,count,old_name=IMAGES[label]
    old=read(BASELINE/old_name)
    check(q.get('pass') is True and q.get('target')=='rdna3','failed/non-RDNA3 image gate')
    check(q.get('rtol')==0 and q.get('comparison')=='strict_less_than','image policy changed')
    check(q.get('absolute_limit',q.get('absolute_logit_limit'))==2e-7,'image logit limit changed')
    check(len(q['results'])==count and case_names(q)==case_names(old),'image case coverage differs from Gen9')
    for r in q['results']:
        check(r['pass'] is True and r.get('checks') and all(c['pass'] for c in r['checks']),'failed required image check')
        for c in r['checks']:
            if 'logits' in c.get('metric','') and 'max_abs' in c:check(0<=c['max_abs']<2e-7,'image logit limit violated')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bin-dir',type=Path,default=CRATE/'target/debug')
    p.add_argument('--test-binary',type=Path,required=True)
    p.add_argument('--stages',nargs='+',choices=STAGES,default=STAGES)
    p.add_argument('--device-id',type=lambda x:int(x,0),default=0x15bf)
    a=p.parse_args();a.frozen=a.frozen.resolve();a.output=a.output.resolve()
    a.bin_dir=a.bin_dir.resolve();a.test_binary=a.test_binary.resolve(strict=True)
    a.output.mkdir(parents=True,exist_ok=False)
    # A diagnostic writer in the retained suite uses this manifest-relative path.
    (CRATE.parent/'target/deepseek-v41-reference').mkdir(parents=True,exist_ok=True)
    frozen=read(a.frozen/'HANDOFF-MANIFEST.json')
    for name,expected in frozen['files'].items():
        expected=expected.get('sha256') if isinstance(expected,dict) else expected
        check(digest(a.frozen/name)==expected,'original frozen oracle changed: '+name)
    binary_hashes={str(path):digest(path) for path in [a.test_binary,*[
        a.bin_dir/(name+'.exe') for name in ['image_text_pipeline_probe','gemma3_vision_probe',
        'gemma4_vision_probe','llava_pipeline_probe','hierarchos-vulkan-transformer-logits']]]}
    records=[]
    def persist():
        complete=len(records)==len(a.stages)
        write(a.output/'qualification.json',dict(schema=1,target='rdna3',scope='complete_green_gen9_multimodal_replay',
            absolute_limit=2e-7,rtol=0,comparison='strict_less_than',
            baseline_sha256=digest(BASELINE/'qualification.json'),
            frozen_manifest_sha256=digest(a.frozen/'HANDOFF-MANIFEST.json'),
            unchanged_frozen_files=len(frozen['files']),binary_sha256=binary_hashes,
            oracle_cpu='AVX2 frozen reference',deepseek_host_width=16,stages=a.stages,
            complete=complete,**{'pass':complete and all(r['pass'] for r in records)},results=records))
    persist()
    for stage in a.stages:
        env={**os.environ,'HIERARCHOS_ATEN_VECTOR_WIDTH':'8'}
        dest=a.output/stage
        print('Starting',stage,flush=True)
        if stage in IMAGES:
            script,fixture,binary,_,_=IMAGES[stage]
            command=[sys.executable,str(CRATE/'validation'/f'{script}.py'),'run','--fixtures',str(a.frozen/fixture),
                '--binary',str(a.bin_dir/(binary+'.exe')),'--output',str(dest),'--target','rdna3',
                '--device-id',hex(a.device_id)]
        elif stage=='text166':
            command=[sys.executable,str(CRATE/'validation/qualify_gen9_architectures.py'),'run',
                '--fixtures',str(a.frozen/'target/gen9-phase3/cpu-fixtures-reproduced'),
                '--binary',str(a.bin_dir/'hierarchos-vulkan-transformer-logits.exe'),'--output',str(dest),
                '--target','rdna3','--expected-device-id',hex(a.device_id)]
        elif stage=='deepseek215':
            env['HIERARCHOS_ATEN_VECTOR_WIDTH']='16'
            command=[sys.executable,str(CRATE/'validation/qualify_rdna3_deepseek_suite.py'),
                '--binary',str(a.test_binary),'--output',str(dest)]
        else:
            env['HIERARCHOS_ATEN_VECTOR_WIDTH']='16'
            command=[sys.executable,str(CRATE/'validation/qualify_deepseek_v41_native_pipeline.py'),
                '--binary',str(a.test_binary),'--output',str(dest),'--target','rdna3','--device-id',hex(a.device_id)]
        result=subprocess.run(command,cwd=CRATE,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        log=result.stdout+result.stderr;(a.output/(stage+'.log')).write_text(log,encoding='utf-8')
        row=dict(stage=stage,command=command,returncode=result.returncode,**{'pass':False})
        try:
            check(result.returncode==0,'stage process failed')
            if stage in IMAGES:image_result(read(dest/'qualification.json'),stage)
            elif stage=='text166':
                q=read(dest/'qualification.json');old=read(BASELINE/'text-qualification.json')
                key=lambda r:(r['family'],r['case'],r['mode'])
                check(q['pass'] and len(q['records'])==166 and {key(r) for r in q['records']}=={key(r) for r in old['records']},
                    'resident/streaming coverage differs from Gen9')
                check(all(r['pass'] and 0<=r['max_abs']<2e-7 for r in q['records']),'text logit gate failed')
            elif stage=='deepseek215':
                q=read(dest/'qualification.json')
                check(q['pass'] and q['verified_passed']==217 and q['complete'],'incomplete DeepSeek suite')
                check(set(q['ignored'])==set(read(BASELINE/'deepseek-suite-qualification.json')['ignored']),'ignored coverage changed')
            else:
                q=read(dest/'qualification.json')
                check(q['pass'] and q['completed_tests']==q['expected_tests']==30,'native V4.1 incomplete')
            check(all(digest(Path(path))==sha for path,sha in binary_hashes.items()),'qualified binary changed during replay')
            row['pass']=True
        except (ValueError,KeyError) as error:row['error']=str(error)
        records.append(row);persist();print(stage,'PASS' if row['pass'] else 'FAIL',row.get('error',''),flush=True)
    check(all(r['pass'] for r in records),'full replay contains failed stages')

if __name__=='__main__':main()
