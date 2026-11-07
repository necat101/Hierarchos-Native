"""Same-binary ABBA native V4.1 hardware-FMA ablation on physical RDNA3.

Keep grouped uploads enabled in both arms. Compare every final logit and
image/adapter digest after two warmups and repeated measured optimizer updates.
The three shared-backbone cases are controls for a native-only specialization.
"""
import argparse,json,math,os,re,statistics,subprocess
from pathlib import Path
from benchmark_deepseek_v41_training import digest,signature

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--steps',type=int,default=7)
    a=p.parse_args()
    if a.rounds<2 or a.rounds%2 or not 3<=a.steps<=100:
        raise ValueError('use an even number of paired rounds >=2 and 3..100 measured updates')
    binary=a.binary.resolve(strict=True);fixtures=a.fixtures.resolve(strict=True)
    a.output.mkdir(parents=True,exist_ok=False);sha=digest(binary)
    names={'native_vit_4x4','native_vit_6x4','llama','gpt2','t5'}
    reference=None;runs=[]
    for round_ in range(a.rounds):
        order=['software','hardware'] if round_%2==0 else ['hardware','software']
        for mode in order:
            name=f'{round_+1}-{mode}';destination=(a.output/(name+'.json')).resolve()
            env={**os.environ,'HIERARCHOS_ATEN_VECTOR_WIDTH':'16',
                'HIERARCHOS_MULTIMODAL_BENCH_OUTPUT':str(destination),
                'HIERARCHOS_MULTIMODAL_BENCH_FIXTURES':str(fixtures),
                'HIERARCHOS_MULTIMODAL_BENCH_STEPS':str(a.steps),
                'HIERARCHOS_VULKAN_RDNA3_V41':'1' if mode=='hardware' else '0'}
            for key in ['HIERARCHOS_V41_INDIVIDUAL_UPLOADS','HIERARCHOS_VULKAN_PROFILE_KERNELS',
                'HIERARCHOS_VULKAN_SUBMISSION_REPORT','HIERARCHOS_V41_NATIVE_DIAGNOSTICS',
                'HIERARCHOS_V41_HC_REPLAY','HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS',
                'HIERARCHOS_VULKAN_FORCE_VENDOR']:
                env.pop(key,None)
            print('Starting',name,flush=True)
            result=subprocess.run([str(binary),'multimodal_training_throughput_benchmark',
                '--ignored','--nocapture','--test-threads=1'],env=env,capture_output=True,
                text=True,encoding='utf-8',errors='replace')
            log=result.stdout+result.stderr;(a.output/(name+'.log')).write_text(log,encoding='utf-8')
            if result.returncode or not re.search(r'test result: ok\. 1 passed; 0 failed; 0 ignored;',log):
                raise ValueError(name+': benchmark failed')
            if digest(binary)!=sha:raise ValueError('benchmark binary changed')
            raw=json.loads(destination.read_text());records=raw['records']
            if raw['warmup']!=2 or raw['iterations']!=a.steps or len(records)!=5 or {r['case'] for r in records}!=names:
                raise ValueError('benchmark coverage changed')
            words={r['case']:signature(r) for r in records}
            if reference is None:reference=words
            if words!=reference:raise ValueError(name+': final logit or parameter words changed')
            for record in records:
                if len(record['samples'])!=a.steps or any(not math.isfinite(s['total_ms']) or s['total_ms']<=0 for s in record['samples']):
                    raise ValueError('invalid timing samples')
            runs.append(dict(round=round_+1,mode=mode,records=records))
            (a.output/'run-status.json').write_text(json.dumps([dict(round=r['round'],mode=r['mode'],word_exact=True) for r in runs],indent=2)+'\n')
            print(name,'word exact after',a.steps+2,'updates',flush=True)
    results=[]
    for case in sorted(names):
        arms={}
        for mode in ['software','hardware']:
            rows=[next(r for r in run['records'] if r['case']==case) for run in runs if run['mode']==mode]
            medians=[statistics.median(s['total_ms'] for s in row['samples']) for row in rows]
            arms[mode]=dict(median_ms=statistics.median(medians),round_medians_ms=medians,
                submissions=sorted({s['submissions'] for row in rows for s in row['samples']}))
        results.append(dict(case=case,**arms,speedup=arms['software']['median_ms']/arms['hardware']['median_ms']))
    receipt=dict(schema=1,**{'pass':True},word_exact=True,order='ABBA',rounds=a.rounds,
        host_vector_width=16,native_private_vector_width=8,grouped_uploads=True,
        measured_steps=a.steps,warmup_steps=2,updates_before_digest=a.steps+2,
        binary_sha256=sha,runner_sha256=digest(Path(__file__)),results=results,
        scope='five frozen tiny training cases; same binary, grouped uploads in both arms; no production-size claim')
    (a.output/'benchmark.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
