"""Replay the separate native AVX512 multimodal track and expanded GPT2 shapes.

CPU oracles stay frozen. Native V4.1 retains its private AVX2 numerical scope
inside the AVX512 host process. Run after building, without concurrent GPU jobs.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from check_rdna3_gpt2_receipts import validate_qualification

CRATE=Path(__file__).resolve().parents[1]
PLANS={
    'shapes':('qualify_rdna3_gpt2_shapes','gpt2-shapes-final-cpu','image_text_pipeline_probe',16),
    'shared':('qualify_shared_image_pipeline','cpu-shared-v2','image_text_pipeline_probe',74),
    'kimi':('qualify_shared_image_pipeline','cpu-kimi','image_text_pipeline_probe',3),
    'seq2seq':('qualify_seq2seq_image_pipeline','cpu-seq2seq','image_text_pipeline_probe',6),
    'gemma3':('qualify_gemma3_pipeline','cpu-gemma3','gemma3_vision_probe',4),
    'gemma4':('qualify_gemma4_pipeline','cpu-gemma4','gemma4_vision_probe',2),
    'llava':('qualify_llava_pipeline','cpu-llava','llava_pipeline_probe',4),
    'bridge':('qualify_deepseek_v41_shared_image','cpu-deepseek-v41-v1',None,2),
}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,default=CRATE/'target/rdna3-port')
    p.add_argument('--bin-dir',type=Path,default=CRATE/'target/debug')
    p.add_argument('--test-binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.fixtures=a.fixtures.resolve();a.bin_dir=a.bin_dir.resolve()
    a.test_binary=a.test_binary.resolve(strict=True);a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    binaries={a.test_binary,*[a.bin_dir/(row[2]+'.exe') for row in PLANS.values() if row[2]]}
    binary_hashes={str(path):digest(path) for path in binaries}
    records=[];env={**os.environ,'HIERARCHOS_ATEN_VECTOR_WIDTH':'16'}
    for key in ['HIERARCHOS_VULKAN_PROFILE_KERNELS','HIERARCHOS_VULKAN_SUBMISSION_REPORT']:
        env.pop(key,None)
    def persist():
        complete=len(records)==len(PLANS)
        q=dict(schema=1,target='rdna3',scope='native_avx512_multimodal_and_expanded_gpt2',
            host_vector_width=16,native_v41_private_vector_width=8,
            absolute_limit=2e-7,rtol=0,comparison='strict_less_than',
            complete=complete,**{'pass':complete and all(r['pass'] for r in records)},
            expected_stages=8,retained_cases=95,expanded_shape_cases=16,
            binary_sha256=binary_hashes,runner_sha256=digest(Path(__file__)),results=records)
        (a.output/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
    persist()
    for name,(script,folder,binary,count) in PLANS.items():
        manifest=a.fixtures/folder/'manifest.json'
        oracle=json.loads(manifest.read_text())
        if oracle['oracle']['cpu']!=('AVX2' if name=='bridge' else 'AVX512'):
            raise ValueError(name+': wrong CPU oracle contract')
        command=[sys.executable,str(CRATE/'validation'/(script+'.py')),'run',
            '--fixtures',str(a.fixtures/folder),'--binary',
            str(a.bin_dir/(binary+'.exe') if binary else a.test_binary),'--output',str(a.output/name)]
        if name!='shapes':command+=['--target','rdna3','--device-id','0x15bf']
        print('Starting native',name,flush=True)
        result=subprocess.run(command,cwd=CRATE,env=env,capture_output=True,text=True,
            encoding='utf-8',errors='replace')
        (a.output/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        row=dict(stage=name,command=command,returncode=result.returncode,
            fixture_manifest_sha256=digest(manifest),expected_cases=count,**{'pass':False})
        try:
            if result.returncode:raise ValueError('qualifier process failed')
            q=json.loads((a.output/name/'qualification.json').read_text())
            validate_qualification(q,count)
            if any(digest(Path(path))!=sha for path,sha in binary_hashes.items()):
                raise ValueError('qualified binary changed')
            row['pass']=True
        except (ValueError,KeyError) as error:row['error']=str(error)
        records.append(row);persist()
        print('native',name,'PASS' if row['pass'] else 'FAIL',row.get('error',''),flush=True)
    if not all(r['pass'] for r in records):raise SystemExit(1)

if __name__=='__main__':main()
