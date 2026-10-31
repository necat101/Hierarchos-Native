"""Run the independent CPU/PEFT oracle through streamed graph parameters.

This records operator/PEFT coverage; post-load conversion alone is not a bounded
initial-loading certificate. Each attempt gets a new directory, preserving all
previous accepted or failed evidence. It never acquires full production weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
FAMILIES=['gpt2','llama','mixtral','qwen3_next','deepseek_v4',
    'phi4_multimodal_text','phi3','kimi_k25_text','kimi_k3_kda','kimi_k3_mla','kimi_k3_text',
    'gpt_oss','smollm3','qwen2_5_gqa','qwen2_5_sliding_tied','qwen3_5_full',
    'qwen3_5_linear','qwen3_5_mixed','qwen3_5_moe','qwen4_exp_linear','qwen4_exp_qsa',
    'qwen4_exp_ple','qwen4_exp_mixed_ple','mistral4','minimax_m3_dense','minimax_m3',
    'gemma4','minimax_m2','gemma3','falcon_h1_gated_after','falcon_h1_gated_before','falcon_h1_silu']

def qualify(family,out):
    fixture=ROOT/'hierarchos-vulkan/.peft-oracle-fixtures/parameter-streaming'/out.name/family
    env=dict(os.environ,HIERARCHOS_PEFT_USE_PREBUILT='1',HIERARCHOS_PEFT_STREAM_PARAMETERS='1',
        HIERARCHOS_PEFT_BIN_DIR=str(out/'bin'),
        HIERARCHOS_PEFT_FIXTURE_ROOT=str(fixture))
    start=time.monotonic()
    with (out/f'{family}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(ROOT/'hierarchos-vulkan/validation/verify_peft_lora_strict.py'),family],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    report_path=fixture/'report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    native_path=fixture/'native-base.json'
    native=json.loads(native_path.read_text()) if native_path.exists() else {}
    return family,dict(exit_code=result.returncode,elapsed_seconds=time.monotonic()-start,fixture_directory=str(fixture),
        streamed_linears=native.get('streamed_linears'),retired_parameter_bytes=native.get('retired_parameter_bytes'),
        source_first_parameters=native.get('source_first_parameters',False),
        loading_peak_buffer_bytes=native.get('loading_peak_buffer_bytes'),
        largest_loading_buffer_bytes=native.get('largest_loading_buffer_bytes'),report=report)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('families',nargs='*',choices=FAMILIES)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    out=ROOT/'target/parameter-streaming'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    (out/'bin').mkdir()
    binaries={}
    for stem in ['hierarchos-vulkan-transformer-logits','transformer_parity']:
        name=stem+('.exe' if os.name=='nt' else '')
        source=ROOT/'hierarchos-vulkan/target/debug'/name
        destination=out/'bin'/name
        shutil.copy2(source,destination)
        binaries[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    results={}
    print('Evidence directory',out,flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(qualify,family,out) for family in (a.families or FAMILIES)]
        for job in as_completed(jobs):
            family,result=job.result();results[family]=result
            print(f"{family}: exit={result['exit_code']} pass={result['report'].get('pass')}",flush=True)
            (out/'qualification.json').write_text(json.dumps(dict(scope='ordinary frozen native linears and vocabulary; application streaming constructor',
                bounded_initial_loading_qualified=False,
                source_first_profiles=sorted(k for k,v in results.items() if v['source_first_parameters']),
                binaries=binaries,strict_atol=2e-7,rtol=0,results=results),indent=2)+'\n')
    return int(any(r['exit_code'] or not r['report'].get('pass') for r in results.values()))

if __name__=='__main__':raise SystemExit(main())
